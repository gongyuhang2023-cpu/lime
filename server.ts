import { Hono, type Context } from "hono";
import { bearerAuth } from "hono/bearer-auth";
import { cors } from "hono/cors";
import { serveStatic } from "hono/deno";
import { HTTPException } from "hono/http-exception";
import { logger } from "hono/logger";
import { verifyKey } from "./key.ts";
import {
	decryptJsonWithSavedKey,
	encryptJson,
	HIAE_ENCRYPTION_HEADER,
	HIAE_ENCRYPTION_VERSION,
	type SecurePayload,
} from "./utils/secure_payload.ts";
import type { Config } from "./utils/config.d.ts";

let userConfig: Config | undefined;

try {
	userConfig = (await import("./user_config.ts")).default;
} catch {
	console.log("使用默认配置");
}

const config = userConfig || (await import("./config.ts")).default;

const { single_ci, commit, getUserData, addUserWord, checkAddUserWord } =
	config.runner;

// ---- 生词记录 ----
// 上游没有任何落盘，重启即失忆（README「现状」明说）。这里补上：
// 同一个词被选够 LIME_LEARN_AFTER 次才收录，避免把每次选择都写成用户词。
// 写的就是启动时读的那个 config.userWordsPath，闭环之后重启不再丢。
const LEARN_AFTER = Number(Deno.env.get("LIME_LEARN_AFTER")) || 3;
const LEARN_MAX_LEN = Number(Deno.env.get("LIME_LEARN_MAX_LEN")) || 12;
const commitCount = new Map<string, number>();
/** 本进程已处理过的词，避免重复查验和重复写盘 */
const learnDone = new Set<string>();

// ---- 输入记录 ----
// LIME_RECORD=<路径> 开启：把上屏文本追加落盘，用于日后扩充词库、
// 生成真实评测语料（Claude 编的语料对 LLM 偏可预测，基线数字会偏乐观）。
//
// 默认关闭，必须显式指定路径 —— 输入法看得见你敲的**一切**，包括密码、
// 私信、身体状况。这个文件是本机明文，不要放进任何会同步或提交的目录。
const RECORD_PATH = Deno.env.get("LIME_RECORD");

async function recordCommit(text: string) {
	if (!RECORD_PATH) return;
	try {
		// 句末断行，产出的文件可直接当评测语料喂 LIME_BENCH_TEXT
		await Deno.writeTextFile(
			RECORD_PATH,
			/[。！？\n]$/.test(text) ? `${text}\n` : text,
			{ append: true },
		);
	} catch (e) {
		console.error("输入记录落盘失败:", e);
	}
}

async function learnFromCommit(text: string) {
	if (!text || text.length > LEARN_MAX_LEN || learnDone.has(text)) return;
	const n = (commitCount.get(text) ?? 0) + 1;
	commitCount.set(text, n);
	if (n < LEARN_AFTER) return;

	learnDone.add(text);
	// addUserWord 内部按 token 序列去重，启动时已加载的词会返回 false
	if (!(await checkAddUserWord(text)) || !addUserWord(text)) return;
	try {
		await Deno.writeTextFile(config.userWordsPath, `${text}\n`, {
			append: true,
		});
		console.log(`记录生词「${text}」（被选 ${n} 次）`);
	} catch (e) {
		console.error("生词落盘失败:", e);
	}
}

function arrayLimtPush<T>(arr: T[], item: T, maxLen: number) {
	arr.push(item);
	if (arr.length <= maxLen) return;
	for (let i = 0; i < arr.length - maxLen; i++) {
		arr.shift();
	}
}

const inputLogMaxLen = 10 ** 5;
export const inputLog: {
	keyDeltaTimes: Array<number>;
	lastKeyTime: number | null;
	ziDeltaTimes: Array<number>;
	lastZiTime: number | null;
	ziCount: number;
	lastCandidates: {
		time: number;
		candidates: string[];
	};
	offsetTimes: Record<number, Array<number>>;
	history: string; // 与直接从模型获取记录不同，模型有上下文限制，这里记录所有输入的文本，供后续微调使用
} = {
	keyDeltaTimes: [],
	lastKeyTime: null,
	ziDeltaTimes: [],
	lastZiTime: null,
	ziCount: 0,
	lastCandidates: {
		time: 0,
		candidates: [],
	},
	offsetTimes: {},
	history: "",
};

try {
	const words = Deno.readTextFileSync(config.userWordsPath)
		.split("\n")
		.filter((w) => w.trim());
	const textEncoder = new TextEncoder();
	for (const [i, w] of words.entries()) {
		// userWordsPath 存的是本人加过 / 从输入里学到的词，给优先权。
		// 批量导入的通用词库走 LIME_BULK_WORDS，不给优先权（见 vouchedTokens）。
		addUserWord(w);
		Deno.stdout.writeSync(
			textEncoder.encode(
				`加载用户词 ${(((i + 1) / words.length) * 100).toFixed(2)}%\r`,
			),
		);
	}
	console.log(`\n加载用户词完成，数量 ${words.length}`);
} catch {
	//
}

// 批量通用词库（可选）：只让这些词出现在候选里，不给同长度优先。
// 实测把两万通用词也当成「用户亲自确认」来抬权重，总选择成本反而从 364
// 涨到 563，比不加词库还差 —— 优先权稀缺才有用。
const bulkPath = Deno.env.get("LIME_BULK_WORDS");
if (bulkPath) {
	try {
		let n = 0;
		for (const w of Deno.readTextFileSync(bulkPath).split("\n")) {
			if (w.trim() && addUserWord(w.trim(), false)) n++;
		}
		console.log(`加载批量词库完成，数量 ${n}`);
	} catch (e) {
		console.error("批量词库加载失败:", e);
	}
}

const app = new Hono();
const api = new Hono();
api.use("/*", async (c, next) => {
	const path = new URL(c.req.url).pathname;
	const isEncryptedInputRequest = c.req.method === "POST" &&
		c.req.header(HIAE_ENCRYPTION_HEADER) === HIAE_ENCRYPTION_VERSION &&
		(path.endsWith("/candidates") || path.endsWith("/commit"));
	if (isEncryptedInputRequest) {
		return next();
	}

	const middleware = bearerAuth({
		verifyToken: (t) => {
			return verifyKey(t);
		},
	});
	return middleware(c, next);
});

api.use("*", logger());

async function readRequestJson<T>(
	c: Context,
): Promise<{ body: T; responseKey?: Uint8Array }> {
	if (c.req.header(HIAE_ENCRYPTION_HEADER) !== HIAE_ENCRYPTION_VERSION) {
		return { body: await c.req.json<T>() };
	}

	const payload = await c.req.json<SecurePayload>();
	const decrypted = await decryptJsonWithSavedKey<T>(payload);
	if (!decrypted) {
		throw new HTTPException(401, { message: "HiAE 请求认证失败" });
	}
	return { body: decrypted.value, responseKey: decrypted.key };
}

function jsonResponse(c: Context, value: unknown, key?: Uint8Array) {
	if (!key) return c.json(value);
	c.header(HIAE_ENCRYPTION_HEADER, HIAE_ENCRYPTION_VERSION);
	return c.json(encryptJson(value, key));
}

api.post("/candidates", async (c) => {
	const { body, responseKey } = await readRequestJson<{ keys?: string }>(c);
	const keys = body.keys || "";

	console.log(keys);
	const time = Date.now();
	if (inputLog.lastKeyTime === null || keys.length === 1) {
		inputLog.lastKeyTime = time;
		inputLog.lastZiTime = time;
	} else {
		arrayLimtPush(
			inputLog.keyDeltaTimes,
			time - inputLog.lastKeyTime,
			inputLogMaxLen,
		);
		inputLog.lastKeyTime = time;
	}

	const pinyinInput = config.key2ZiInd(keys);
	const result = await single_ci(pinyinInput);

	if (result.candidates.length <= 1) {
		inputLog.lastZiTime = null;
	} else
		inputLog.lastCandidates = {
			time,
			candidates: result.candidates.map((c) => c.word),
		};

	return jsonResponse(c, result, responseKey);
});

api.post("/commit", async (c) => {
	try {
		const { body, responseKey } = await readRequestJson<{
			text?: string;
			new?: boolean;
			update?: boolean;
		}>(c);
		const text = body.text || "";
		const isNew = body.new ?? true;
		const shouldUpdate = body.update ?? false;

		if (!text) {
			throw new HTTPException(400, { message: "未提供文本内容" });
		}

		const newT = await commit(text, shouldUpdate, isNew);

		if (isNew) {
			if (inputLog.lastZiTime !== null)
				arrayLimtPush(
					inputLog.ziDeltaTimes,
					(Date.now() - inputLog.lastZiTime) / text.length,
					inputLogMaxLen,
				);
			inputLog.lastZiTime = null;
			inputLog.lastKeyTime = null;
			inputLog.ziCount += text.length;
			inputLog.history += text;
			await recordCommit(text);
			await learnFromCommit(text);
		}
		{
			const offset = inputLog.lastCandidates.candidates.indexOf(newT ?? "");
			if (offset !== -1 && inputLog.lastCandidates.time !== 0) {
				const time = Date.now();
				const ofts = inputLog.offsetTimes[offset] || [];
				arrayLimtPush(
					ofts,
					time - inputLog.lastCandidates.time,
					inputLogMaxLen,
				);
				inputLog.offsetTimes[offset] = ofts;
			}
			inputLog.lastCandidates = {
				time: 0,
				candidates: [],
			};
		}

		return jsonResponse(
			c,
			{
				message: "文本提交成功",
			},
			responseKey,
		);
	} catch (error) {
		if (error instanceof HTTPException) throw error;
		console.error("提交文本失败:", error);
		throw new HTTPException(400, { message: "请求数据格式错误" });
	}
});

api.get("/userdata", (c) => {
	return c.json(getUserData());
});

api.get("/inputlog", (c) => {
	return c.json(inputLog);
});

api.post("/learntext", async (c) => {
	const body = await c.req.text();
	await commit(body, true, true);
	return c.json({
		message: "文本提交成功",
	});
});

try {
	Deno.statSync("./interface/dist");
} catch {
	console.log(
		"没有构建前端，一些服务器页面可能不显示（不影响输入法），如果需要，运行：\ndeno run install_interface\ndeno run build_interface\n然后重启服务器",
	);
}

app.use(
	"*",
	cors({
		origin: "*",
	}),
);

app.use("/*", serveStatic({ root: "./interface/dist" }));

app.route("/api", api);

app.post("/candidates", (c) => {
	return api.fetch(c.req.raw);
});

app.post("/commit", (c) => {
	return api.fetch(c.req.raw);
});

export default app;

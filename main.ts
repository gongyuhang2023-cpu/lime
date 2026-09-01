import path from "node:path";
import { fileURLToPath } from "node:url";
import {
	getLlama,
	type LlamaContext,
	type LlamaContextSequence,
	type LlamaModel,
	type Token,
} from "node-llama-cpp";
import type { ZiIndAndKey, ZiIndL } from "./key_map/zi_ind.ts";
import { ziid_in_ziid } from "./utils/ziind_in_ziind.ts";

type ZiIndFunc = (zici: string) => string[][];

export type Candidate = {
	word: string;
	score: number;
	pinyin: string[];
	remainkeys: string[];
	preedit: string;
	consumedkeys: number;
	/** 该候选来自用户词表（用户显式登记过），排序时同长度优先 */
	userWord?: boolean;
};

export type Result = {
	candidates: Candidate[];
};

export type UserData = {
	words: Record<number, Array<number>>;
	context: Array<{ t: string; token: Token }>;
};

const __dirname = path.dirname(fileURLToPath(import.meta.url));

class Lock {
	private pm: Promise<void> | null = null;

	async acquire() {
		if (this.pm) await this.pm;
	}

	async lock() {
		await this.acquire();
		const p = Promise.withResolvers<void>();
		this.pm = p.promise;

		return {
			release: () => {
				p.resolve();
			},
		};
	}
}

export async function loadModel(op?: {
	modelPath?: string;
	contextSize?: number;
}) {
	const modelPath =
		op?.modelPath ??
		path.join(__dirname, "../Qwen3-0.6B-GGUF/Qwen3-0.6B-IQ4_XS.gguf");

	const llama = await getLlama({
		// 上游写死 gpu:false。实测 RTX 5070 上走 Vulkan，基准耗时从约 40 分钟
		// 降到约 10 分钟，故本 fork 默认交给 node-llama-cpp 自动选后端
		//（无可用 GPU 时它会自己退回 CPU）。LIME_GPU=0 可强制纯 CPU。
		gpu: Deno.env.get("LIME_GPU") === "0" ? false : "auto",
	});

	console.log("加载模型", modelPath);

	const model = await llama.loadModel({
		modelPath: modelPath,
	});
	const context = await model.createContext({
		contextSize: { max: op?.contextSize ?? 4096 },
	});
	console.log("加载完成");

	return { model, context };
}

export async function initLIME(
	op: Parameters<typeof loadModel>[0] & {
		ziInd: { trans: ZiIndFunc; allSymbol: Set<string> };
		omitContext?: boolean;
		afterReSort?: Array<AfterReSortFunc>;
	},
) {
	const { model, context } = await loadModel(op);
	const lime = new LIME({
		model,
		context,
		ziInd: op.ziInd,
		omitContext: op.omitContext,
		afterReSort: op.afterReSort,
	});
	await lime.init_ctx();
	return lime;
}

type ExToken = Token | number;
export type AfterReSortFunc = (candidates: Candidate[]) => Candidate[];

export class LIME {
	model: LlamaModel;
	context: LlamaContext;
	sequence: LlamaContextSequence;

	token_pinyin_map: Map<number, Array<Array<string>>> = new Map();
	first_pinyin_token = new Map<string, Set<number>>();
	unIndexedZi = new Map<string, Set<string>>();

	private pre_context = "下面的内容主题多样";
	last_context_data = { context: "" };
	private userTokens = new Map<ExToken, Array<Token>>();
	private userTokensFirstIndex = new Map<Token, Set<ExToken>>();
	/** 已登记用户词的 token 序列，用于去重 */
	private userWordKeys = new Set<string>();
	private tokenIndex = 0;

	/**
	 * 候选表头部最多允许几个长候选，之后让位给短候选；0 = 不限制（上游行为）。
	 * 默认 3：实测 3 优于 5，总选择成本（偏移加权）从 391 降到 364。
	 */
	private longHead = Number(Deno.env.get("LIME_LONG_HEAD") ?? 3) || 0;

	private last_result: Map<ExToken, number> | undefined;
	/** 长句补全，记录拼音和token对 */
	private longSentenceCache: {
		py: ZiIndL;
		matchPY: ZiIndAndKey[];
		token: ExToken[];
		nextResult: Map<ExToken, number>;
	}[] = [];
	private lastCommitOffset = 0;

	private afterReSort: Array<AfterReSortFunc>;

	private modelEvalLock = new Lock();

	/** 运行时触发 */
	private max_count = 4000;
	/** 空闲时触发，大任务，所以提前打算，触发值小一点 */
	private smallerMaxCount = 3500;
	private rm_count = 20;
	private omitContext = new deBounce(1000 * 10, async () => {
		await this.modelEvalLock.acquire();
		const { release } = await this.modelEvalLock.lock();
		await this.tryOmitContext();
		release();
	});

	constructor({
		model,
		context,
		ziInd,
		omitContext,
		afterReSort,
	}: {
		model: LlamaModel;
		context: LlamaContext;
		ziInd: { trans: ZiIndFunc; allSymbol: Set<string> };
		omitContext?: boolean;
		afterReSort?: Array<AfterReSortFunc>;
	}) {
		this.model = model;
		this.context = context;
		this.sequence = context.getSequence();

		this.max_count = context.contextSize - 64;
		this.smallerMaxCount =
			this.max_count - Math.min(512, Math.floor(this.max_count * 0.1));
		this.rm_count = Math.min(
			this.smallerMaxCount - 64,
			Math.floor(this.smallerMaxCount * 0.2),
		);
		if (!omitContext) this.omitContext.cancel();

		this.afterReSort = afterReSort ?? [];

		console.log("创建拼音索引");

		const { trans, allSymbol } = ziInd;

		let max = 0;
		for (const token_id of model.iterateAllTokens()) {
			max = Math.max(max, token_id);
		}
		this.tokenIndex = max + 1;

		// todo 先解码字，再遍历所有token建立索引
		for (const token_id of model.iterateAllTokens()) {
			const token = model.detokenize([token_id]);
			if (!token) continue;
			const pinyins = trans(token);
			allSymbol.delete(token);
			if (pinyins.length) {
				this.token_pinyin_map.set(token_id, pinyins);
				for (const fp of pinyins[0]) {
					const s = this.first_pinyin_token.get(fp) ?? new Set();
					s.add(token_id);
					this.first_pinyin_token.set(fp, s);
				}
			}
		}
		for (const zi of allSymbol) {
			const pys = trans(zi);
			if (pys.length === 1) {
				for (const py of pys[0]) {
					const s = this.unIndexedZi.get(py) ?? new Set();
					s.add(zi);
					this.unIndexedZi.set(py, s);
				}
			}
		}
		if (allSymbol.size > 0) {
			console.log(
				"以下字未直接建立拼音索引:",
				Array.from(allSymbol).slice(0, 10).join(" "),
				"等",
			);
		}
	}

	private tryOmitContext = async () => {
		if (this.sequence.contextTokens.length <= this.smallerMaxCount) {
			return;
		}
		const oldTokenLen = this.sequence.contextTokens.length;

		// 输入分词可能有些情况不是像分词器那样的切分，会影响模型性能，这里重编码分词
		const oldTokens = this.sequence.contextTokens.slice();
		const oldText = this.model.detokenize(oldTokens);
		const newTokens = this.model
			.tokenizer(oldText)
			.slice(-Math.max(this.smallerMaxCount - this.rm_count, 1));

		await this.sequence.clearHistory();
		await this.sequence.controlledEvaluate([
			...newTokens.slice(0, -1),
			[
				// biome-ignore lint/style/noNonNullAssertion: none
				newTokens.at(-1)!,
				{
					generateNext: {
						probabilities: true,
						options: {
							topK: Infinity,
						},
					},
				},
			],
		]);
		this.lastCommitOffset = this.sequence.contextTokens.length;
		console.log(
			`已优化上下文 ${oldTokenLen}->${this.sequence.contextTokens.length}`,
		);
	};

	// 快速从最新删除上下文，保证后面操作不爆
	private fastTryOmitContext = async (buffer = 64) => {
		const maxCount = this.max_count - Math.max(buffer, 64);

		if (this.sequence.contextTokens.length <= maxCount) {
			return;
		}
		await this.sequence.eraseContextTokenRanges([
			{
				start: maxCount,
				end: this.sequence.contextTokens.length,
			},
		]);
		this.lastCommitOffset = this.sequence.contextTokens.length;
	};

	commit = async (text: string, update = false, newT = true) => {
		let new_text = "";
		let nt = newT;

		this.longSentenceCache = [];

		if (update) {
			if (text.startsWith(this.last_context_data.context)) {
				new_text = text.slice(this.last_context_data.context.length);
				this.last_context_data.context = text;
			} else {
				new_text = text;
				nt = true;
			}
		}
		if (nt) {
			this.last_context_data.context = "";
			if (update === false) {
				new_text = text;
			}
		}
		if (!new_text) return;

		// todo shift context

		const to_run = this.model.tokenizer(new_text);
		if (to_run.length === 0) return;

		const pre = to_run.slice(0, -1);
		const last = to_run[to_run.length - 1];
		const { release } = await this.modelEvalLock.lock();
		// 强制commit耗时的部分为异步执行，避免请求阻塞
		(async () => {
			await this.fastTryOmitContext(pre.length + 1);
			// todo 根据缓存判断，比如长句实际上已经近似提交了
			await this.sequence.eraseContextTokenRanges([
				{
					start: this.lastCommitOffset,
					end: this.sequence.contextTokens.length,
				},
			]);
			const res = await this.sequence.controlledEvaluate([
				...pre,
				[
					last,
					{
						generateNext: {
							probabilities: true,
						},
					},
				],
			]);
			this.last_result = res.at(-1)?.next.probabilities; // todo 如果在自定义中某个tk值比较大，那尝试多运行一步
			if (this.last_result) this.applyUserTokenProbs(this.last_result);
			this.lastCommitOffset = this.sequence.contextTokens.length;
			release();
		})();

		this.omitContext.reset();

		return new_text;
	};

	reset_context = async () => {
		await this.modelEvalLock.acquire();
		this.last_context_data.context = "";
		this.userTokens.clear();
		this.userWordKeys.clear();
		await this.sequence.clearHistory();
		await this.init_ctx();
	};

	getEvalResult = async () => {
		await this.modelEvalLock.acquire();
		return this.last_result;
	};

	/**
	 * 用户词是虚拟 token，模型分布里没有它，上游三处都直接填 0（注释写着「临时」）。
	 * 代价是任何按概率排序的逻辑对用户词全部失效 —— 实测 addUserWord("冰灯") 之后
	 * 打 bingdeng，冰灯拿着 score 0 输给了 score 0.5 的「并等」。
	 *
	 * 这里改用首个真实 token 的概率作为估计：用户词的后续字是用户亲自登记过的，
	 * 把不确定性压在首字上是合理近似，而且不需要额外前向。填进 last_result 后
	 * 会和其他候选一起过 filterByPinyin 的归一化，量纲自然对齐。
	 */
	private applyUserTokenProbs = (r: Map<ExToken, number>) => {
		for (const [id, ts] of this.userTokens) {
			const head = ts[0];
			r.set(id, head === undefined ? 0 : (r.get(head) ?? 0));
		}
	};

	exTokens = (tokens: ExToken[]) => {
		const tks = tokens.flatMap((i) => this.userTokens.get(i) ?? [i as Token]);
		return tks;
	};
	detoken = (tokens: ExToken[]) => {
		return this.model.detokenize(this.exTokens(tokens));
	};

	checkAddUserWord = async (w: string) => {
		const ts = this.model.tokenizer(w);
		if (ts.length <= 1) return false;
		if (ts.some((i) => !this.token_pinyin_map.has(i))) return false;
		// 外部自己去重

		// todo 根据模型值来看，有些词模型可以很轻松预测出来，不需要加入用户词库
		return true;
	};

	/**
	 * 把一个虚拟 token 注册进各张索引。addUserWord 与 loadUserData 共用：
	 * 只往 userTokens 里塞映射是不够的，不建拼音索引的话这个词永远不会进候选集。
	 */
	private registerUserToken = (token_id: ExToken, ts: Array<Token>) => {
		this.userTokens.set(token_id, ts);
		this.userWordKeys.add(ts.join(","));

		const findex = this.userTokensFirstIndex.get(ts[0]) ?? new Set();
		findex.add(token_id);
		this.userTokensFirstIndex.set(ts[0], findex);

		const pys = ts.flatMap((i) => this.token_pinyin_map.get(i) || []);

		this.token_pinyin_map.set(token_id, pys);
		for (const fp of pys[0]) {
			const s = this.first_pinyin_token.get(fp) ?? new Set();
			s.add(token_id);
			this.first_pinyin_token.set(fp, s);
		}

		if (this.last_result) {
			this.last_result.set(token_id, this.last_result.get(ts[0]) ?? 0);
		}
	};

	addUserWord = (w: string) => {
		const ts = this.model.tokenizer(w);
		if (ts.length === 0) return false;

		// 上游把去重交给调用方（checkAddUserWord 的注释），但重复登记会生成两个
		// 虚拟 token，候选里就会出现两个一模一样的词。这里按 token 序列兜底。
		if (this.userWordKeys.has(ts.join(","))) return false;

		this.registerUserToken(this.tokenIndex++, ts);

		return true;
	};

	single_ci = async (pinyin_input: ZiIndL): Promise<Result> => {
		if (pinyin_input.length === 0 || pinyin_input[0].length === 0) {
			return { candidates: [] };
		}

		if (!this.last_result) {
			return { candidates: [] };
		}

		const c: Candidate[] = [];

		await this.modelEvalLock.acquire();
		await this.fastTryOmitContext(pinyin_input.length);

		const filterByPinyin = (
			pinyin_input: ZiIndL,
			last_result: Map<ExToken, number>,
		) => {
			const new_last_result = new Map<
				ExToken,
				{ py: ZiIndAndKey[]; prob: number; token: string }
			>();
			let scoreSum = 0;
			const ftokenid = new Set<number>();
			for (const firstPinyin of pinyin_input[0]) {
				const s = this.first_pinyin_token.get(firstPinyin.ind) ?? new Set();
				for (const tokenid of s) ftokenid.add(tokenid);
			}

			for (const [token_id, token_prob] of last_result) {
				if (!ftokenid.has(token_id)) continue;
				const token = this.detoken([token_id]);
				if (!token) continue;
				if (["\t", "\n", " "].includes(token[0])) continue;

				const token_pinyin_dy = this.token_pinyin_map.get(token_id);

				if (!token_pinyin_dy) continue;

				const token_pinyin = ziid_in_ziid(pinyin_input, token_pinyin_dy);
				if (!token_pinyin) continue;
				if (token === token_pinyin[0].ind) continue; // 排除部分英文
				new_last_result.set(token_id, {
					py: token_pinyin,
					prob: token_prob,
					token: this.detoken([token_id]),
				});
				scoreSum += token_prob;
			}
			for (const v of new_last_result.values()) {
				v.prob /= scoreSum;
			}

			// 长词优先
			const first = new_last_result.values().next().value;
			if ((first?.prob ?? 0) < 0.9) {
				const n = new Map() as typeof new_last_result;

				let maxLen = 0;
				let longToken: typeof first;
				let longTokenId: ExToken | undefined;
				for (const [k, v] of new_last_result) {
					if (v.py.length > maxLen) {
						maxLen = v.py.length;
						longToken = v;
						longTokenId = k;
					}
				}
				let reOrder = false;
				if (maxLen > 1 && longToken && longTokenId) {
					n.set(longTokenId, longToken);
					reOrder = true;
				}
				for (const [k, v] of new_last_result) {
					if (reOrder === false || k !== longTokenId) {
						n.set(k, v);
					}
				}
				return n;
			}

			return new_last_result;
		};
		const new_last_result = filterByPinyin(pinyin_input, this.last_result);

		// 首个候选补全为长句
		await (async () => {
			const token_id = new_last_result.keys().next().value;
			if (!token_id) return;
			const _r = new_last_result.get(token_id);
			if (!_r) return;
			const { py: token_pinyin, prob: token_prob } = _r;

			if (pinyin_input.length === token_pinyin.length) {
				this.longSentenceCache = [];
				return;
			}

			let sameCacheLen = 0;
			let pyIndex = 0;

			for (const [i, cache] of this.longSentenceCache.entries()) {
				const cpyl = cache.py;
				const inputPyl = pinyin_input.slice(pyIndex, pyIndex + cpyl.length);
				if (JSON.stringify(cpyl) !== JSON.stringify(inputPyl)) {
					break;
				}
				sameCacheLen = i + 1;
				pyIndex += cpyl.length;
			}
			let sameCache = this.longSentenceCache.slice(0, sameCacheLen);
			let rmpyx = pinyin_input.slice(
				sameCache.flatMap((i) => i.matchPY).length,
			);
			{
				const lc = this.longSentenceCache.slice(0, sameCacheLen);
				const last = lc.at(-1);
				if (last) {
					const npy = last.py.concat(rmpyx);
					const lastlast = lc.at(-2)?.nextResult || this.last_result;
					if (lastlast) {
						const f = filterByPinyin(npy, lastlast);
						const first = f.entries().next().value;
						if (first) {
							if (first[0] !== last.token.at(-1)) {
								sameCacheLen--;
								sameCache = this.longSentenceCache.slice(0, sameCacheLen);
								rmpyx = pinyin_input.slice(
									sameCache.flatMap((i) => i.matchPY).length,
								);
							}
						}
					} else {
						console.warn("no lastlast");
					}
				}
			}

			// 必须先把用户词的虚拟 token 展开成真实 token：sequence.contextTokens
			// 里只有真实 token，不展开则长度和内容都对不上 —— 下面的比对会一直
			// 报「长句缓存不匹配」，而 eraseContextTokenRanges 会按错误的长度
			// 裁剪，直接破坏上下文。（上游此处的 todo 指的就是这件事。）
			// 用户词以前分数恒为 0、进不了长句缓存，所以这个 bug 一直没显形；
			// 给用户词补上真实分数之后，预置 79 个领域词跑一遍基准就报了 590 次。
			const cacheTokens = this.exTokens(sameCache.flatMap((i) => i.token));
			if (
				this.sequence.contextTokens
					.slice(
						this.lastCommitOffset,
						this.lastCommitOffset + cacheTokens.length,
					)
					.join(",") !== cacheTokens.join(",")
			) {
				console.error("长句缓存不匹配");
			}
			await this.sequence.eraseContextTokenRanges([
				{
					start: this.lastCommitOffset + cacheTokens.length,
					end: this.sequence.contextTokens.length,
				},
			]);
			if (
				cacheTokens.at(-1) &&
				cacheTokens.at(-1) !== this.sequence.contextTokens.at(-1)
			) {
				console.error("erase error");
			}

			this.longSentenceCache = this.longSentenceCache.slice(0, sameCacheLen);

			let prob = token_prob;
			const tklppy: ZiIndAndKey[] = [...sameCache.flatMap((i) => i.matchPY)];
			const tkl: ExToken[] = [...cacheTokens];

			const select = (op: {
				py: ZiIndL;
				matchPY: ZiIndAndKey[];
				token: ExToken[];
				nextResult: Map<ExToken, number>;
			}) => {
				tklppy.push(...op.matchPY);
				tkl.push(...op.token);
				rmpyx = pinyin_input.slice(tklppy.length);

				this.longSentenceCache.push({
					py: op.py,
					matchPY: op.matchPY,
					token: op.token,
					nextResult: op.nextResult,
				});
			};

			const addToken = async (token: ExToken) => {
				const tks = this.exTokens([token]);
				const r =
					(
						await this.sequence.controlledEvaluate([
							...tks.slice(0, -1),
							[
								// biome-ignore lint/style/noNonNullAssertion: none
								tks.at(-1)!,
								{
									generateNext: {
										probabilities: true,
									},
								},
							],
						])
					).at(-1)?.next.probabilities || new Map<ExToken, number>(); // todo
				this.applyUserTokenProbs(r);
				return r;
			};

			if (this.longSentenceCache.length === 0)
				select({
					token: [token_id],
					matchPY: token_pinyin,
					py: pinyin_input.slice(0, token_pinyin.length),
					nextResult: await addToken(token_id),
				});

			const l = rmpyx.length;

			await this.fastTryOmitContext(l);

			for (let _i = 0; _i < Math.min(l, 4); _i++) {
				const next = this.longSentenceCache.at(-1)?.nextResult;
				if (!next) {
					console.log("no next");
					break;
				}
				const f = filterByPinyin(rmpyx, next);
				if (f.size > 0) {
					const first = f.entries().next().value;
					if (first) {
						prob *= first[1].prob;
						const tp = first[1];
						select({
							token: [first[0]],
							matchPY: tp.py,
							py: pinyin_input.slice(
								tklppy.length,
								tklppy.length + tp.py.length,
							),
							nextResult: await addToken(first[0]),
						});
						if (rmpyx.length === 0) {
							break;
						}
					}
				}
			}

			if (tkl.length > 1) {
				c.push({
					pinyin: tklppy.map((v) => v.ind),
					score: prob,
					word: this.detoken(tkl),
					remainkeys: rmpyx.map((v) => v[0].ind),
					preedit:
						tklppy.map((v) => v.preeditShow).join(" ") +
						(rmpyx.length ? " " : ""),
					consumedkeys: tklppy.map((v) => v.key).join("").length,
				});
			}
		})();

		// 常规
		for (const [
			token_id,
			{ py: token_pinyin, prob: token_prob, token },
		] of new_last_result) {
			const rmpy = pinyin_input.slice(token_pinyin.length).map((v) => v[0].ind);
			c.push({
				pinyin: token_pinyin.map((v) => v.ind),
				score: token_prob,
				word: token,
				remainkeys: rmpy,
				preedit:
					token_pinyin.map((v) => v.preeditShow).join(" ") +
					(rmpy.length ? " " : ""),
				consumedkeys: token_pinyin.map((v) => v.key).join("").length,
				userWord: this.userTokens.has(token_id),
			});
		}

		for (const py of pinyin_input[0]) {
			const unIndexSet = this.unIndexedZi.get(py.ind);
			if (unIndexSet) {
				for (const zi of unIndexSet) {
					c.push({
						pinyin: [py.ind],
						score: 0.0001,
						word: zi,
						remainkeys: pinyin_input.slice(1).map((v) => v[0].ind),
						preedit: py.preeditShow + (pinyin_input.length > 1 ? " " : ""),
						consumedkeys: py.key.length,
					});
				}
			}
		}

		// 长词优先；同等长度时，用户显式登记过的词优先于模型即兴拼出的同音串
		// —— 这是用户词典本来的语义，批量导入词库时它们彼此之间再按概率排序。
		c.sort(
			(a, b) =>
				b.pinyin.length - a.pinyin.length ||
				Number(b.userWord ?? false) - Number(a.userWord ?? false),
		);

		// 长候选的首屏配额。
		//
		// 上游只按拼音长度排序，于是敲一整句拼音时，候选表头部会被一大串等长的
		// 错误整句猜测占满，正确的短词被挤到很后面 —— 实测最惨一例，正确的单字
		// 排在第 54 位（page_size 5，等于翻 11 页）。
		//
		// 先试过给长候选设「每音节平均置信度」闸门，无效：那些垃圾长句的置信度
		// 高于 0.65，模型不是不自信，是自信地错，这个杠杆撬不动它。
		// 改为限制头部允许多少个长候选，保证短候选在首屏就拿得到，
		// 把猜错时的代价从翻十几页压到翻一页。
		let tc = c;
		if (this.longHead > 0) {
			const long = c.filter((x) => x.pinyin.length > 1);
			if (long.length > this.longHead) {
				tc = long
					.slice(0, this.longHead)
					.concat(
						c.filter((x) => x.pinyin.length <= 1),
						long.slice(this.longHead),
					);
			}
		}
		for (const f of this.afterReSort) {
			tc = f(tc);
		}

		this.omitContext.reset();

		if (c.length === 0) {
			console.log("is empty");
		}
		return { candidates: tc };
	};

	init_ctx = async () => {
		const prompt = this.pre_context;
		const tokens = this.model.tokenizer(prompt);
		const [pre, last] = [tokens.slice(0, -1), tokens.at(-1)];
		if (last === undefined) {
			throw "初始token不够";
		}
		const x = await this.sequence.controlledEvaluate([
			...pre,
			[
				last,
				{
					generateNext: {
						probabilities: true,
						options: {
							topK: Infinity,
						},
					},
				},
			],
		]);
		this.last_result = x.at(-1)?.next.probabilities;
		this.lastCommitOffset = this.sequence.contextTokens.length;
	};

	getUserData = () => {
		return {
			words: Object.fromEntries(this.userTokens),
			context: this.sequence.contextTokens.map((t) => ({
				t: this.model.detokenize([t]) || "",
				token: t,
			})),
		} as UserData;
	};
	/**
	 * 恢复 getUserData() 导出的数据。
	 *
	 * 「记忆」是两层，恢复难度差一个量级：
	 *  - 用户词：纯数据，重建索引即可（registerUserToken）。
	 *  - 上下文：是 llama.cpp 的 KV cache，属于模型中间状态、不是数据，
	 *    只能把 token 序列重新喂一遍让它重建。长度受 contextSize 限制，
	 *    超出部分由既有的 tryOmitContext 裁掉。
	 */
	loadUserData = async (data: UserData) => {
		if (this.userTokens.size > 0) {
			console.log("已存在用户数据");
			return;
		}
		for (const [k, v] of Object.entries(data.words)) {
			const id = Number(k);
			this.registerUserToken(id, v as Token[]);
			// 虚拟 id 必须跳过已恢复的，否则下一个 addUserWord 会复用 id 覆盖掉旧词
			this.tokenIndex = Math.max(this.tokenIndex, id + 1);
		}

		const tokens = data.context.map((i) => i.token);
		const last = tokens.at(-1);
		if (last === undefined) return;

		await this.modelEvalLock.acquire();
		const { release } = await this.modelEvalLock.lock();
		try {
			await this.sequence.clearHistory();
			const res = await this.sequence.controlledEvaluate([
				...tokens.slice(0, -1),
				[last, { generateNext: { probabilities: true, options: { topK: Infinity } } }],
			]);
			this.last_result = res.at(-1)?.next.probabilities;
			if (this.last_result) this.applyUserTokenProbs(this.last_result);
			this.lastCommitOffset = this.sequence.contextTokens.length;
		} finally {
			release();
		}
		console.log(
			`恢复用户数据：${this.userTokens.size} 个用户词，${tokens.length} 个上下文 token`,
		);
	};
}

class deBounce {
	private timeout: ReturnType<typeof setTimeout> | null = null;
	private delay: number;
	private fun = () => {};
	private cancelled = false;
	constructor(delay: number, fun: () => void) {
		this.delay = delay;
		this.fun = fun;
	}

	reset() {
		if (this.timeout) clearTimeout(this.timeout);
		if (this.cancelled) return;
		this.timeout = setTimeout(() => {
			this.fun();
		}, this.delay);
	}

	cancel() {
		if (this.timeout) clearTimeout(this.timeout);
		this.cancelled = true;
	}
}

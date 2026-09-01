import path from "node:path";
import { fileURLToPath } from "node:url";
import type { Config } from "../utils/config.d.ts";
import { get_dict } from "../key_map/rime_dict.ts";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

let userConfig: Config | undefined;

try {
	userConfig = (await import("../user_config.ts")).default;
} catch {
	console.log("使用默认配置");
}

const config = userConfig || (await import("../config.ts")).default;

const filePath = path.join(__dirname, "preload_word.txt");

const { checkAddUserWord } = config.runner;

const words: string[] = [];

if (Deno.args[0] === "rime") {
	const p = Deno.args[1];
	const d = get_dict(p);
	for (const w of d) {
		const word = w.split("\t")[0].trim();
		const value = Number(w.split("\t")[2]?.trim() || "0");
		if (word && value > 5000) words.push(word);
	}
} else if (Deno.args[0] === "text") {
	// 从一份中文语料里提取高频多字词，配合 LIME_RECORD 记下来的真实输入使用：
	//   deno run -A userword/preload_word.ts text <语料.txt> [最少出现次数,默认2]
	// 基准实测「补领域词库」是收益最大的一项，而记录下来的真实用词比手写的准。
	const p = Deno.args[1];
	const minCount = Number(Deno.args[2]) || 2;
	const raw = Deno.readTextFileSync(p);
	const seg = new Intl.Segmenter("zh-Hans", { granularity: "word" });
	const freq = new Map<string, number>();
	for (const t of seg.segment(raw)) {
		const w = t.segment.trim();
		// 只要纯汉字的多字词：单字模型自己会预测，标点数字英文不进词库
		if (!t.isWordLike || w.length < 2 || !/^[一-鿿]+$/.test(w)) continue;
		freq.set(w, (freq.get(w) ?? 0) + 1);
	}
	for (const [w, n] of freq) if (n >= minCount) words.push(w);
	words.sort((a, b) => (freq.get(b) ?? 0) - (freq.get(a) ?? 0));
	console.log(`从语料提取到 ${words.length} 个词（出现 >=${minCount} 次）`);
}

const oldWords = new Set<string>();
try {
	for (const x of Deno.readTextFileSync(filePath).split("\n")) {
		if (x.trim()) oldWords.add(x.trim());
	}
} catch {
	// ignore
}

const textEncoder = new TextEncoder();
for (const [i, w] of words.entries()) {
	const res = await checkAddUserWord(w);
	if (res) oldWords.add(w);
	Deno.stdout.writeSync(
		textEncoder.encode(
			`预加载用户词 ${(((i + 1) / words.length) * 100).toFixed(2)}%\r`,
		),
	);
}
console.log(`\n保存完毕`);

Deno.writeTextFileSync(
	path.join(__dirname, "preload_word.txt"),
	Array.from(oldWords).join("\n"),
);
console.log("预加载用户词完成，数量", oldWords.size);

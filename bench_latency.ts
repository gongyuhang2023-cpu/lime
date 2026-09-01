// 逐次按键延迟基准：输入法的硬约束是「每敲一个键要多久出候选」，
// 而 test_text.ts 只统计偏移，不体现引擎自身耗时。
import { load_pinyin } from "./key_map/pinyin/gen_zi_pinyin.ts";
import { keys_to_pinyin } from "./key_map/pinyin/keys_to_pinyin.ts";
import { initLIME } from "./main.ts";

const modelPath = Deno.env.get("LIME_BENCH_MODEL");
const t0 = performance.now();
const { single_ci, commit, addUserWord } = await initLIME({
	ziInd: load_pinyin(),
	...(modelPath ? { modelPath } : {}),
});
const loadMs = performance.now() - t0;

// 每个用户词都会往 last_result 里加一个虚拟 token，而 filterByPinyin 每次
// 按键都要遍历它 —— 词库规模直接压在按键延迟上，必须一起量
const t1 = performance.now();
let nWords = 0;
const wordsFile = Deno.env.get("LIME_BENCH_USERWORDS");
if (wordsFile) {
	for (const w of Deno.readTextFileSync(wordsFile).split("\n")) {
		if (w.trim() && addUserWord(w.trim())) nWords++;
	}
}
const wordMs = performance.now() - t1;

// 模拟真实输入：一句一句敲，每敲一个字母量一次
const sentences = [
	"zhizhinamilishimuqianhesuanyaowudisonglingyu",
	"womenzhepiyongweiliukongxinpianzhibei",
	"yanyangpeiyangxiangdeyanghanliangchuanganqi",
	"yizhixingbiwoyuxiangdedagerenduo",
];
const lat: number[] = [];
for (const s of sentences) {
	for (let i = 1; i <= s.length; i++) {
		const t = performance.now();
		await single_ci(keys_to_pinyin(s.slice(0, i)));
		lat.push(performance.now() - t);
	}
	await commit("。");
}
lat.sort((a, b) => a - b);
const q = (p: number) => lat[Math.floor(lat.length * p)].toFixed(0);
console.log(
	JSON.stringify({
		model: (modelPath ?? "默认0.6B").split(/[\/]/).pop(),
		用户词: nWords,
		启动秒: (loadMs / 1000).toFixed(1),
		装词秒: (wordMs / 1000).toFixed(1),
		按键数: lat.length,
		p50: q(0.5),
		p90: q(0.9),
		p99: q(0.99),
		最大: lat.at(-1)?.toFixed(0),
	}),
);

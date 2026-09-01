import path from "node:path";
import { fileURLToPath } from "node:url";
import { load_pinyin } from "./key_map/pinyin/gen_zi_pinyin.ts";
import { keys_to_pinyin } from "./key_map/pinyin/keys_to_pinyin.ts";
import { initLIME } from "./main.ts";
import type { Config } from "./utils/config.d.ts";
import { resortFeq } from "./tools/resort_feq.ts";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const config: Config = {
	runner: await initLIME({
		ziInd: load_pinyin(),
		// 上游默认 0.6B。实测同为 IQ4_XS 量化时，1.7B 把总选择成本降了 32%
		// （偏移加权 575→391），首选命中率 38.5%→47.1%，故本 fork 默认用 1.7B。
		// 换回 0.6B 或换别的模型：复制本文件为 user_config.ts 后改这一行。
		modelPath: path.join(
			__dirname,
			"../Qwen3-1.7B-GGUF/Qwen3-1.7B-IQ4_XS.gguf",
		),
		omitContext: true,
		afterReSort: [
			resortFeq(
				Deno.readTextFileSync(path.join(__dirname, "assets/hanzi/top2500.txt"))
					.split("\n")
					.filter((w) => w.trim()),
				{ index: 4 },
			),
		],
	}),
	key2ZiInd: (key: string) =>
		keys_to_pinyin(key, {
			shuangpin: false,
			fuzzy: {
				initial: {
					c: "ch",
					z: "zh",
					s: "sh",
					ch: "c",
					zh: "z",
					sh: "s",
				},
				final: {
					an: "ang",
					ang: "an",
					en: "eng",
					eng: "en",
					in: "ing",
					ing: "in",
					uan: "uang",
					uang: "uan",
				},
			},
		}),
	userWordsPath: path.join(__dirname, "userword/preload_word.txt"),
};

export default config;

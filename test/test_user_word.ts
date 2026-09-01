import { assertEquals } from "@std/assert";
import { load_pinyin } from "../key_map/pinyin/gen_zi_pinyin.ts";
import { keys_to_pinyin } from "../key_map/pinyin/keys_to_pinyin.ts";
import { initLIME } from "../main.ts";

const { commit, single_ci, addUserWord, getUserData } = await initLIME({
	ziInd: load_pinyin(),
});

Deno.test("组词", async () => {
	addUserWord("冰灯");
	const r = await single_ci(keys_to_pinyin("bingdeng"));
	console.log(r.candidates.slice(0, 5));
	assertEquals(r.candidates[0].word, "冰灯");
});

Deno.test("智能组词", async () => {
	await commit("冰灯");
	await commit("是");
	await commit("流行于");
	const nr = await single_ci(keys_to_pinyin("vsgobz", { shuangpin: "自然码" }));
	console.log(nr.candidates.slice(0, 5));
});

Deno.test("用户数据往返恢复", async () => {
	// 原实例：登记用户词 + 累积一点上下文
	addUserWord("冰灯");
	await commit("冰灯是");
	const data = getUserData();

	// 新实例：只靠导出的数据恢复，应当能打出用户词
	const fresh = await initLIME({ ziInd: load_pinyin() });
	await fresh.loadUserData(data);

	const r = await fresh.single_ci(keys_to_pinyin("bingdeng"));
	console.log("恢复后候选:", r.candidates.slice(0, 3).map((i) => i.word));
	assertEquals(r.candidates[0].word, "冰灯");
});

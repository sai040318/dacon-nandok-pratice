# test_kor_obfus_preprocess.py
# -*- coding: utf-8 -*-
"""
kor_obfus_preprocess.py 용 간단 테스트

실행:
    python -m unittest -v test_kor_obfus_preprocess.py

Colab:
    !python -m unittest -v test_kor_obfus_preprocess.py
"""

import unittest

from kor_obfus_preprocess import (
    is_hangul_syllable,
    decompose_char,
    compose_char,
    align_text_pair_for_jamo,
    collect_jamo_change_stats,
    build_axis_rules,
    JamoRules,
    restore_with_jamo_rules,
)


class TestHangulBasics(unittest.TestCase):
    def test_is_hangul_syllable(self):
        self.assertTrue(is_hangul_syllable("가"))
        self.assertTrue(is_hangul_syllable("힣"))
        self.assertFalse(is_hangul_syllable("A"))
        self.assertFalse(is_hangul_syllable("ㄱ"))  # 자모 단독은 False

    def test_decompose_compose_roundtrip(self):
        for ch in ["가", "값", "힣", "편", "안", "히", "캠"]:
            d = decompose_char(ch)
            self.assertIsNotNone(d)
            self.assertEqual(compose_char(*d), ch)

    def test_align_text_pair_for_jamo(self):
        inp = "가 나 다"
        out = "가나다라마"
        a, b = align_text_pair_for_jamo(inp, out)
        # 공백 제거 후 짧은 쪽 길이에 맞춰 잘린다
        self.assertEqual(a, "가나다")
        self.assertEqual(b, "가나다")
        self.assertEqual(len(a), len(b))


class TestStatsRulesRestore(unittest.TestCase):
    def test_collect_stats_and_build_rule(self):
        # '거'(ㄱ,ㅓ,'') -> '겨'(ㄱ,ㅕ,'') 패턴만 존재하는 작은 데이터
        pairs = [("거", "겨")] * 10 + [("가", "가")] * 10
        stats = collect_jamo_change_stats(pairs)

        # input 중성 통계: ㅓ가 10번, ㅏ가 10번 등장
        self.assertEqual(stats.V_totals["ㅓ"], 10)
        self.assertEqual(stats.V_totals["ㅏ"], 10)
        # 변화: ㅓ -> ㅕ 가 10번
        self.assertEqual(stats.V_changes[("ㅓ", "ㅕ")], 10)

        # 룰 생성: ㅓ는 항상 ㅕ로 바뀌었으므로 ㅓ->ㅕ 룰이 생성되어야 함
        V_rule = build_axis_rules(
            stats.V_changes, stats.V_totals,
            min_total=1, min_top_prob=0.8, allow_identity=False
        )
        self.assertEqual(V_rule.get("ㅓ"), "ㅕ")

    def test_restore_with_rules(self):
        # V축에만 ㅓ->ㅕ 룰을 준다
        rules = JamoRules(L_rule={}, V_rule={"ㅓ": "ㅕ"}, T_rule={})
        self.assertEqual(restore_with_jamo_rules("거", rules), "겨")
        self.assertEqual(restore_with_jamo_rules("거!", rules), "겨!")  # 문장부호 유지
        self.assertEqual(restore_with_jamo_rules("가", rules), "가")    # 영향 없음


if __name__ == "__main__":
    unittest.main()

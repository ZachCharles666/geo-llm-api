from geo_core import geo_cot_stage1, geo_cot_stage2
from geo_cot_parser import stage1_md_to_json, stage2_text_to_blueprint, check_blueprint_consistency

user_question = "请帮我为一家制造业CRM软件写一个 GEO 友好的产品介绍大纲"
brand_brief = "超兔CRM是一款专为制造业中小工厂设计的客户与订单管理系统，帮助企业打通线索、报价、订单、发货与售后全流程数据，让老板随时看到销售漏斗和订单利润情况。"
must_expose = ""
expo_hint = ""

# 1) 跑 Stage1
s1_md, _ = geo_cot_stage1(
    user_question=user_question,
    brand_brief=brand_brief,
    must_expose=must_expose,
    expo_hint=expo_hint,
    model_ui="groq",
)

s1_json = stage1_md_to_json(
    md_text=s1_md,
    user_question=user_question,
    brand_brief=brand_brief,
    must_expose=must_expose,
    expo_hint=expo_hint,
)

# 2) 跑 Stage2
s2_md, _ = geo_cot_stage2(
    user_question=user_question,
    brand_brief=brand_brief,
    must_expose=must_expose,
    stage1_md=s1_md,
    model_ui="groq",
    expo_hint=expo_hint,
)

bp = stage2_text_to_blueprint(s2_md, user_question=user_question, brand_brief=brand_brief)

print("=== Blueprint keys ===", bp.keys())
print("chains:", len(bp.get("chains", [])))
print("titles:", len(bp.get("titles", [])))
print("abstracts:", len(bp.get("abstracts", [])))
print("routes:", bp.get("routes"))

diag = check_blueprint_consistency(bp)
print("=== CC 对齐诊断 ===")
print(diag)

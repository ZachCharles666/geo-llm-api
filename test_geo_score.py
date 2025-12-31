from geo_evaluator import geo_score_pipeline

if __name__ == "__main__":
    q = "请帮我为一家制造业CRM软件写一个 GEO 友好的产品介绍大纲"
    src = "这是原始文案，比较啰嗦，也不太 GEO 友好，可以随便写几句用来测试。"
    opt = "这是已经过 GEO 工具优化后的版本，用来测试评分用。"

    for tier in ["free", "alpha", "pro"]:
        print(f"\n=== user_tier = {tier} ===")
        res = geo_score_pipeline(
            user_question=q,
            source_text=src,
            rewritten_text=opt,
            user_tier=tier,
        )
        print(res)

from typing import Any, Dict, List

from .metrics import rouge_ja, LanguageDetector


def compute_score(results: Dict[str, List[Any]]) -> Dict[str, float]:
    lang_detect = LanguageDetector()
    res_dict = rouge_ja(refs=results["answer"], preds=results["prediction"])
    # detect Japanese by fasttext and replace empty string if it's not Ja
    if lang_detect:
        preds = []
        for answer, pred in zip(results["answer"], results["prediction"]):
            # if answer is English, pass
            if lang_detect(answer).get("__label__ja", 0.0) >= 0.5:
                res = lang_detect(pred)
                if res.get("__label__ja", 0.0) < 0.5:
                    pred = ""
            preds.append(pred)
        res_dict_ja = rouge_ja(refs=results["answer"], preds=preds)
        res_dict_ja = {f"{k}_ja": v for k, v in res_dict_ja.items()}
        res_dict.update(res_dict_ja)
    return res_dict


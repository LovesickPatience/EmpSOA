from tqdm import tqdm
from copy import deepcopy
# from tensorboardX import SummaryWriter
from torch.nn.init import xavier_uniform_

import torch
from src.utils import config
from src.utils.common import set_seed
from src.models.EmpSOA.model import EmpSOA
from src.utils.data.loader import prepare_data_seq
from src.models.common import evaluate, count_parameters

def make_model(vocab, dec_num):
    is_eval = config.test
    model = EmpSOA(
        vocab,
        decoder_number=dec_num,
        is_eval=is_eval,
        model_file_path=config.model_path if is_eval else None,
    )

    model.to(config.device)

    # Intialization
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n != "embedding.lut.weight" and config.pretrain_emb):
            xavier_uniform_(p)

    print("# PARAMETERS", count_parameters(model))

    return model


def train(model, train_set, dev_set):
    last_epoch = 0
    try:
        model.train()
        best_ppl = 1000
        patient = 0
        weights_best = deepcopy(model.state_dict())
        for epoch in range(config.epochs):
            last_epoch = epoch
            pbar = tqdm(enumerate(train_set), total=len(train_set))
            for n_iter, batch in pbar:
                loss, ppl, bce, acc, _, _ = model.train_one_batch(
                    batch, n_iter
                )
                pbar.set_description(
                    "loss:{:.4f} ppl:{:.1f}".format(loss, ppl)
                )

            model.eval()
            model.epoch = epoch
            loss_val, ppl_val, bce_val, acc_val, _ = evaluate(
                model, dev_set, ty="valid", max_dec_step=50
            )
            model.train()

            if ppl_val <= best_ppl:
                best_ppl = ppl_val
                patient = 0
                if config.save:
                    model.save_model(best_ppl, epoch)
                weights_best = deepcopy(model.state_dict())
            else:
                patient += 1
            if patient > 2:
                break

    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")
        model.save_model(best_ppl, last_epoch)
        weights_best = deepcopy(model.state_dict())

    return weights_best


def test(model, test_set):
    model.eval()
    model.is_eval = True
    loss_test, ppl_test, bce_test, acc_test, results = evaluate(
        model, test_set, ty="test", max_dec_step=50
    )
    if config.save_decode:
        file_summary = config.save_path + "/results.txt"
        with open(file_summary, "w") as f:
            f.write("EVAL\tLoss\tPPL\tAccuracy\n")
            f.write(
                "{}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(
                    loss_test, ppl_test, bce_test, acc_test
                )
            )
            for r in results:
                f.write(r)


def eval_valid_metrics(model, dev_set):
    model.eval()
    print("VALIDATION METRICS ....")
    (
        loss_val,
        ppl_val,
        bce_val,
        acc_val,
        _,
        dist1,
        dist2,
        avg_len,
        refs,
        hyps,
    ) = evaluate(
        model,
        dev_set,
        ty="valid",
        max_dec_step=50,
        return_texts=True,
        calc_gen_metrics=True,
    )
    print(
        "VALID: {:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}".format(
            loss_val,
            ppl_val,
            bce_val,
            acc_val,
            dist1 or 0.0,
            dist2 or 0.0,
            avg_len or 0.0,
        )
    )
    bert_f1 = None
    try:
        from bert_score import score as bert_score
    except ImportError:
        print("BERTScore not available. Install bert-score to enable it.")
    else:
        _, _, f1 = bert_score(
            hyps,
            refs,
            lang=config.bertscore_lang,
            model_type=config.bertscore_model,
            verbose=False,
        )
        bert_f1 = f1.mean().item()
        print("BERTScore F1: {:.4f}".format(bert_f1))

    file_summary = config.save_path + "/valid_metrics.txt"
    with open(file_summary, "w") as f:
        f.write("Loss\tPPL\tBCE\tAcc\tDist1\tDist2\tAvgLen\tBERTScoreF1\n")
        f.write(
            "{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}\t{}\n".format(
                loss_val,
                ppl_val,
                bce_val,
                acc_val,
                dist1 or 0.0,
                dist2 or 0.0,
                avg_len or 0.0,
                "" if bert_f1 is None else "{:.4f}".format(bert_f1),
            )
        )

def main():
    set_seed()  # for reproducibility

    train_set, dev_set, test_set, vocab, dec_num = prepare_data_seq(
        batch_size=config.batch_size
    )

    model = make_model(vocab, dec_num)

    if config.test:
        test(model, test_set)
    else:
        weights_best = train(model, train_set, dev_set)
        model.epoch = 1
        model.load_state_dict({name: weights_best[name] for name in weights_best})
        if config.eval_valid_metrics:
            eval_valid_metrics(model, dev_set)
        test(model, test_set)


if __name__ == "__main__":
    # python main.py --model cem --cuda --csk_feature --dis_emo_cog --save_decode
    # python main.py --model cem --cuda --csk_feature --dis_emo_cog --wo_dis_sel_gen --save_decode
    # python main.py --model cem --cuda --csk_feature --dis_emo_cog --wo_dis_sel_gen --wo_dis_sel_reg --save_decode
    # python main.py --model cem --cuda --csk_feature --dis_emo_cog --wo_dis_sel_gen --wo_dis_sel_reg --wo_dis_sel_awa --save_decode
    # python main.py --model cem --cuda --csk_feature --dis_emo_cog --wo_dis_sel_oth --save_decode
    # python main.py --model cem --cuda --csk_feature --dis_emo_cog --only_user --save_decode
    # python main.py --model cem --cuda --csk_feature --dis_emo_cog --only_agent --save_decode
    main()

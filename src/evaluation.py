import matplotlib.pyplot as plt
import pandas as pd
from src.utils import get_the_main_path

NAME_LOGS = "lstm_next_word_predictor"
VERSION = "16"

def create_logger_path():
    main_path = get_the_main_path()
    log_dir = main_path / "logs" / NAME_LOGS /f"version_{VERSION}"
    return log_dir


def plot_metrics(metrics: pd.DataFrame):
    save_path = get_the_main_path() /"results" /f"version_{VERSION}"
    if not save_path.exists():
        save_path.mkdir(parents=True)

    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)
    df_metrics[["train_loss", "val_loss"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="Loss"
    )
    plt.savefig(save_path/"loss.png")

    df_metrics[["train_acc", "valid_acc"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="ACC"
    )
    plt.savefig(save_path/"acc.png")
    plt.show()




def run():
    log_dir = create_logger_path()
    metrics = pd.read_csv(log_dir/"metrics.csv")
    plot_metrics(metrics)

if __name__ == "__main__":
    run()
from src.dataloader import TextDataModule
from src.model import LSTMNextWordPredictor
from src.utils import get_the_main_path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from warnings import filterwarnings
import torch
filterwarnings("ignore")

NUM_EPOCHS = 60
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
HIDDEN_SIZE = 256
EMBEDDING_DIM = 200
NUM_LAYERS = 2
LIMIT = 250
NUM_WORKERS = 0


def run():
    torch.set_float32_matmul_precision('medium')

    main_path = get_the_main_path()
    data_path = main_path / "data" / "final"/ "ted_speech_clean.csv"
    save_dir = main_path / "logs"
    save_model_dir = main_path / "models"
    save_model_path = save_model_dir / "lstm_next_word_predictor.ckpt"
    checkpoint_dir = main_path / "checkpoints"

    data_module = TextDataModule(
        data_path=data_path,
        batch_size=BATCH_SIZE,
        sequence_length=20,
        vocab_size=10000,
        limit=LIMIT,
        num_workers=NUM_WORKERS
    )


    data_module.setup()
    vocab_size = data_module.get_vocab_size()

    # Crear el modelo
    lstm_model = LSTMNextWordPredictor(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        learning_rate=LEARNING_RATE
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='lstm-next-word-{epoch:02d}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss',
        mode='min'
    )

    progress_bar = TQDMProgressBar(refresh_rate=10)

    # Logger
    logger = CSVLogger(save_dir=save_dir, name="lstm_next_word_predictor")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        callbacks=[checkpoint_callback, progress_bar],
        accelerator="auto",  # Usa GPUs o TPUs si están disponibles
        devices="auto",  # Usa todos los GPUs/TPUs disponibles si es aplicable
        logger=logger,
        deterministic=False,
        log_every_n_steps=10
    )

    # Entrenamiento
    trainer.fit(lstm_model, datamodule=data_module)

    # Guardar el modelo
    trainer.save_checkpoint(save_model_path)

    # Evaluación en el conjunto de prueba
    test_result = trainer.test(lstm_model, datamodule=data_module)
    print(f"Resultado del test: {test_result}")


if __name__ == '__main__':
    run()
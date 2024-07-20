from src.model import LSTMNextWordPredictor
from src.dataset import TextDataset
from src.utils import get_the_main_path
import random
import torch

LIMIT = 250


def data_path():
    main_path = get_the_main_path()
    return main_path / "data" / "final" / "ted_speech_clean.csv"


def generate_text(model, start_sequence, max_length=5):
    model.eval()
    current_sequence = start_sequence
    generated_sequence = start_sequence.copy()

    for _ in range(max_length):
        with torch.no_grad():
            input_tensor = torch.tensor([current_sequence]).to(model.device)
            output = model(input_tensor)
            next_word_idx = output[0, -1, :].argmax().item()

        generated_sequence.append(next_word_idx)
        current_sequence = current_sequence[1:] + [next_word_idx]

    return generated_sequence


def load_model():
    # Cargar el modelo
    checkpoint_path = get_the_main_path()/"models"/"lstm_next_word_predictor.ckpt"
    if not checkpoint_path.exists():
        raise FileNotFoundError("El archivo del checkpoint no existe")

    model = LSTMNextWordPredictor.load_from_checkpoint(checkpoint_path)
    model.eval()

    dataset = TextDataset(
        data_path=data_path(),
        sequence_length=50,
        vocab_size=10000,
        limit=LIMIT
    )

    model.set_vocab(dataset.token_to_idx, dataset.idx_to_token)
    # Verificar que el vocabulario se ha establecido correctamente
    print("Tamaño del vocabulario:", len(model.token_to_idx))

    return model

def run(model,sequence, length):


    start_sequence = [model.token_to_idx.get(word.lower(), model.token_to_idx['<UNK>']) for word in sequence.split()]
    generated_indices = generate_text(model, start_sequence, max_length=length)
    generated_text = [model.idx_to_token[idx] for idx in generated_indices]
    print(" ".join(generated_text))

if __name__ == "__main__":
    random_seed = random.randint(0, 1000)
    torch.manual_seed(random_seed)
    model = load_model()
    while True:
        sequence = input("Ingrese la secuencia de palabras: ")
        run(model,sequence, 5)
        print("\n")



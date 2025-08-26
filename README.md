# BlossomTuneMLX-LM: Federated Fine-Tuning of LLMs on Apple Silicon


## Empowering Decentralized & Efficient LLM Customization on Mac

BlossomTuneMLX-LM is an open-source project designed to enable Federated Supervised Fine-Tuning (SFT) of Small Language Models (SLMs), specifically optimized for the Apple Silicon (M1/M2/M3/M4) architecture.

This project adapts the core principles of the original BlossomTuneLLM, replacing the PyTorch and NVIDIA-based backend with Apple's native MLX framework and the powerful mlx-lm library.

The result is a lightweight, efficient, and Docker-free solution for federated learning on macOS devices.

## Why BlossomTune-MLX-LM:

In an era where large language models demand immense computational resources, BlossomTune-MLX-LM offers a powerful alternative for:

- **Decentralization & Privacy-First AI**: Train LLMs collaboratively across multiple Macs without centralizing sensitive data.

- **Native Apple Silicon Performance**: Leverage the full power of the Unified Memory and Neural Engine on M-series chips for efficient LoRA/DoRA fine-tuning.

- **Accessibility for the Apple Ecosystem**: Provides an accessible framework for developers, students, and researchers in the Apple ecosystem to build specialized, privacy-first models, so get ready for a LAN party like it's 1999 again.

- **Customization & Flexibility**: Offers streamlined customization of fine-tuning parameters, target layers, and includes centralized dataset partitioning from Hugging Face with flexible pre-processing.

## Key Features:

- **Federated Supervised Fine-Tuning (SFT)**: Leverages the Flower framework to facilitate federated learning for SLMs.
- **Native Apple Silicon Performance**: Engineered from the ground up to run natively on macOS, using mlx-lm for highly efficient LoRA fine-tuning.
- **Enhanced Customization**: Easily configure fine-tuning parameters, learning rates (with cosine annealing), and LoRA settings.
- **Flexible Data Handling**: Supports centralized dataset partitioning from Hugging Face and dynamic prompt/completion templating for diverse datasets.
- **Apache-2.0 Licensed**: Open and permissive for broad use and collaboration.

## Getting Started

BlossomTuneMLX-LM is designed to run natively on macOS without requiring Docker or other containerization tools.

### Prerequisites

- A Mac with Apple Silicon (M1, M2, M3, M4 series).
- Python 3.10 or newer.
- We recommend using `uv` for fast package management, but `pip` will also work.

First, clone the repository and install the dependencies:

```
git clone https://github.com/ethicalabs-ai/BlossomTuneLLM-MLX.git
cd BlossomTuneLLM-MLX
```

#### Install dependencies using uv (recommended)

```
uv sync
```

#### Or, install using pip

```
pip install -e .
```

## Running Federated Training

You can run BlossomTuneMLX in two primary modes: a simple simulation on a single machine, or a true federated setup across multiple machines on your local network.


### Scenario 1: Running a Simulation on a Single Machine

This is the quickest way to test the entire federated learning pipeline. It starts a server and simulates multiple clients on your local machine. However, due to the limited amount of memory available for the simulation, this works only with Small LMs (135M).

#### Start the Simulation
    
Run the following command in your terminal:

```
flwr run . local-simulation
```

#### Customize the Run (Optional)

You can override any configuration parameter from the pyproject.toml file directly from the command line using a run-config.

For example, to change the model and the number of LoRA layers and server rounds:

```
 uv run flwr run . local-simulation --run-config="train.lora_layers=6 num_server_rounds=20"
 ```

The fine-tuned global model adapters will be saved in the `./results/` directory.

### Scenario 2: Running on Multiple Machines (Real Federated Setup)

Here, we'll set up a true federated network with one machine acting as the server (Superlink) and other machines as clients (Supernodes).

Example Setup:

- Server (Superlink): Mac Mini M4 (e.g., IP address 192.168.1.100)
- Client 1 (Supernode): MacBook Air M1 8 GB
- Client 2 (Supernode): MacBook Air M4 16 GB

#### Step 1: Start the Server (on the Mac Mini)

Find the local IP address of your server machine. You can find this in `System Settings > Network`.

On the Mac Mini, run the following command to start the Flower Superlink.

```
uv run flower-superlink --insecure
```

The server will start and wait for clients to connect.

#### Step 2: Start the Clients (on each MacBook Air)

On each client machine, you will start a Flower Supernode that connects to the server. Ensure the `partition-id` is unique for each client.

##### On MacBook Air 1 (Client 1)

Run this command, replacing 192.168.1.100 with your server's IP address. This client gets the first data partition (`partition-id=0`).

```
# Note: num-partitions should equal the total number of total clients (2 in this case)
uv run flower-supernode --insecure --superlink="192.168.1.100:9092" --node-config="partition-id=0 num-partitions=2"
```

##### On MacBook Air 2 (Client 2)

Run the same command, but with partition-id=1.

```
uv run flower-supernode --insecure --superlink="192.168.1.100:9092" --node-config="partition-id=1 num-partitions=2"
```

You should see messages on the server terminal indicating that the clients have connected.

#### Step 3: Start the Federated Training Run

Finally, from the server machine, you can start the actual training run.

The following command tells the server to orchestrate the training across all connected clients:

```
uv run flwr run . local-deployment --run-config="train.seq_length=512 model.name='lmstudio-community/gemma-3-270m-it-MLX-4bit' num_server_rounds=25"
```

The server will now coordinate the rounds of training. Each client will load its data partition, train locally, and send its updated adapter back to the server for aggregation.

The final global adapters will be saved in the ./results/ directory on the machine where you ran the flwr run command.

## Contributing

We welcome contributions from the community! Feel free to open issues, submit pull requests, or join discussions.

## License

BlossomTuneMLX is released under the Apache-2.0 License.

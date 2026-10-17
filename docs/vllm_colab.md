# Running the agent on vLLM with a free Google Colab GPU

vLLM is the model server many companies run on their own GPU machines. Colab's free tier gives you a T4 GPU, which is enough to serve a small model and run the whole eval on it.

Everything runs inside Colab (the model server, the business API and the agent), so nothing needs to be exposed to the internet.

> Colab and vLLM change often. If a step fails, the error in `vllm.log` usually says why (most often: out of GPU memory, or a version conflict after install). The vLLM flags below come from the vLLM tool-calling docs for Qwen2.5 models.

## Before you start

Push this project to a GitHub repository (public is simplest for cloning in Colab).

## Steps

Open [colab.research.google.com](https://colab.research.google.com), create a new notebook, then choose **Runtime → Change runtime type → T4 GPU**. Run each block below in its own cell.

**1. Get the code and install** (installing vLLM takes a few minutes):

```python
!git clone https://github.com/YOUR-USERNAME/invoice-triage-agent.git
%cd invoice-triage-agent
!pip install -q -r requirements.txt vllm
```

If Colab asks you to restart the session after installing, do it, then run `%cd invoice-triage-agent` again before continuing.

**2. Start vLLM and the business API in the background:**

```python
import subprocess

vllm = subprocess.Popen(
    "vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 8192 "
    "--gpu-memory-utilization 0.85 --enable-auto-tool-choice --tool-call-parser hermes "
    "> vllm.log 2>&1",
    shell=True,
)
api = subprocess.Popen("python -m api.app > api.log 2>&1", shell=True)
```

- `--dtype half`: the T4 doesn't support bfloat16, so use float16.
- `--enable-auto-tool-choice --tool-call-parser hermes`: turns on tool calling for Qwen2.5 models. Without these the model can't call tools.

**3. Wait until the model is loaded** (usually 2 to 5 minutes the first time, because it downloads the model):

```python
import time, requests

for _ in range(120):
    try:
        if requests.get("http://localhost:8000/v1/models", timeout=2).ok:
            print("vLLM is ready")
            break
    except requests.RequestException:
        pass
    time.sleep(5)
else:
    print(open("vllm.log").read()[-3000:])
```

**4. Run the agent and score it:**

```python
!python -m agent.run --provider vllm --model Qwen/Qwen2.5-1.5B-Instruct --name vllm-qwen1.5b --limit 5
```

If those 5 look reasonable, run all 105 items:

```python
!python -m agent.run --provider vllm --model Qwen/Qwen2.5-1.5B-Instruct --name vllm-qwen1.5b
!python -m evals.evaluate results/vllm-qwen1.5b.jsonl
```

**5. Download the results** so you can add them to your repo and compare them with your local runs:

```python
from google.colab import files
files.download("results/vllm-qwen1.5b.jsonl")
```

Put the file in your local `results/` folder, then `python -m evals.evaluate results/*.jsonl` compares every run in one table.

## Going further

- A T4 has 16 GB of memory. `Qwen/Qwen2.5-3B-Instruct` usually fits too; 7B models generally need a bigger GPU or a quantized version.
- Compare speed: vLLM on a GPU should be much faster per item than Ollama on a laptop CPU. That difference is a good line for your README.
- Results are saved after every item. If the run stops while the Colab session is still alive, add `--resume` to the run command to finish only the remaining items (if vLLM or the API died too, repeat steps 2 and 3 first). Free sessions eventually end and Colab deletes their files, so download your results file as soon as the run finishes.

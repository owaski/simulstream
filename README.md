## Installation

The package can be installed with `pip install -e .`.
To use processors based on HuggingFace, use `pip install -e .[hf]` or after the above installation,
install the additional required packages.
Please notice that each custom speech processor may have additional requirements.
For instance, for the [fama_hf_sliding_window_retranslation.yaml](config/fama_hf_sliding_window_retranslation.yaml) example,
you also need to install:

```bash
pip install sentencepiece
```

## Usage

To run the websocker server, use:

```bash
simulstream_server --speech-processor-config $YOUR_SPEECH_PROCESSING_CONFIG
```

by default, this command will take the server configuration from [server.yaml](config/server.yaml).
You can specify a different config file with the option `--server-config`.
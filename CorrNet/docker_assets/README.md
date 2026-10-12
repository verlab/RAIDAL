For CUDA==13.X, run `./create_container_cuda13.sh`

For CUDA==12.X, run `./create_container_cuda12.sh`

You can check your CUDA version with

```
nvidia-smi
```

Regardless of which script you run, the first time you enter the container you must run:

```bash
cd /model/
./setup.sh
```

Now activate the conda env with:

```bash
conda activate modelvenv
```

and you're ready to go!
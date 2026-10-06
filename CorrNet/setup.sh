source /miniconda/etc/profile.d/conda.sh && conda activate modelvenv && pip install -r requirements.txt

# get the code and install
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install . --no-build-isolation
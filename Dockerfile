FROM ubuntu:24.10


ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"


# Install wget to fetch Miniconda
RUN apt-get update && \
    apt-get install -y wget git build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Install Miniconda on x86 or ARM platforms
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh


RUN conda install -y python=3.11 && \
    conda clean -a

RUN bash -c "source /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate"

RUN conda install cmake ninja

RUN pip install mkl-static mkl-include

RUN git config --global http.postBuffer 104857600 && \
    git clone --recursive --depth 1 https://github.com/pytorch/pytorch && \
    cd pytorch && \
    git submodule sync && \
    git submodule update --init --recursive

RUN cd pytorch && \
    pip install -r requirements.txt

RUN cd pytorch && \
    USE_FBGEMM=0 MAX_JOBS=1 python setup.py install

RUN pip install scikit-learn pandas

WORKDIR /app
ENV PYTHONPATH "${PYTHONPATH}:/app/./"
COPY src ./src
COPY data ./data
CMD ["python", "src/train.py"]
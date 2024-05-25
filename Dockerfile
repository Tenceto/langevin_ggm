FROM nvidia/cuda:12.3.0-runtime-ubuntu20.04

# Avoid stuck build due to user prompt
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
	apt-get install --no-install-recommends -y python3 python3-pip python3-wheel python3-dev build-essential && \
	apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update -y && apt install software-properties-common -y && add-apt-repository ppa:git-core/ppa -y && \
	apt update -y && \
	apt-get install git -y

# liblapack-dev libopenblas-dev are required to build skggm
# texlive-fonts-recommended texlive-fonts-extra dvipng cm-super are required for matplotlib to render LaTeX
RUN apt install liblapack-dev libopenblas-dev texlive-fonts-recommended texlive-fonts-extra dvipng cm-super -y

# CMD [ "/bin/bash" ]

COPY . /app
WORKDIR /app
# Install one by one, since skggm depends on Cython to be already installed
RUN cat requirements.txt | xargs -n 1 pip install

# Make skggm compatible with the last version of scikit-learn
RUN sed -i "s|sklearn.utils.testing|sklearn.utils._testing|g" /usr/local/lib/python3.8/dist-packages/inverse_covariance/quic_graph_lasso.py
RUN sed -i "s|sklearn.externals.joblib|joblib|g" /usr/local/lib/python3.8/dist-packages/inverse_covariance/quic_graph_lasso.py
RUN sed -i "s|sklearn.externals.joblib|joblib|g" /usr/local/lib/python3.8/dist-packages/inverse_covariance/model_average.py

ENV DEVCONTAINER=true
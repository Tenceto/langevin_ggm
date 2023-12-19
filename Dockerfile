FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Necessary to build skggm
RUN sudo apt install liblapack-dev libopenblas-dev

# Clone the repo
RUN git clone https://github.com/Tenceto/langevin_ggm

# Install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install LaTeX
RUN sudo apt install texlive-fonts-recommended texlive-fonts-extra
RUN sudo apt install dvipng cm-super

# Make skggm compatible with current sklearn
RUN sed -i 's/sklearn.utils.testing/sklearn.utils._testing/g' /.local/lib/python3.8/site-packages/inverse_covariance/quic_graph_lasso.py
RUN sed -i 's/sklearn.externals.joblib/joblib/g' /.local/lib/python3.8/site-packages/inverse_covariance/quic_graph_lasso.py
RUN sed -i 's/sklearn.externals.joblib/joblib/g' /.local/lib/python3.8/site-packages/inverse_covariance/model_average.py
# Remove the automatic verbosity
RUN sed -i 's/_cpu_map(fun, param_grid, n_jobs, verbose=False)/_cpu_map(fun, param_grid, n_jobs, verbose=True)/g' /.local/lib/python3.8/site-packages/inverse_covariance/model_average.py
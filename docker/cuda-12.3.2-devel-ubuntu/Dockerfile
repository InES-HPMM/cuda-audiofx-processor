FROM nvidia/cuda:12.3.2-devel-ubuntu22.04
WORKDIR /app
RUN apt update && \
    apt install --no-install-recommends -y build-essential python3 python3-pip

RUN pip install cmake

RUN apt clean && rm -rf /var/lib/apt/lists/* 

# COPY requirements.txt .
# RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . .
# CMD ["whatever", "you", "want"
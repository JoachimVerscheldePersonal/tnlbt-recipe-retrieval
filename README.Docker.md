
### Step 2: Build the Docker Image

We use Docker to create a reproducible environment. The `Dockerfile` in this repository uses **PyTorch 2.4.1 with CUDA 11.8** and configures the environment to be optimized for GPU support.

You can either build the image manually or use Docker Compose, depending on your preferred approach.

#### Option 1: Build the Image Manually

To manually build the Docker image, run:

```bash
docker build -t tnlbt-recipe-retrieval .
```

#### Option 2: Use Docker Compose

Alternatively, you can use Docker Compose to build and manage the service. Make sure Docker Compose is installed and run:

```bash
docker-compose build
```

### Step 3: Run the Docker Container

Once the image is built, you can start a container to interact with the project. You have two options for running the container, depending on whether you want to use plain Docker commands or Docker Compose.

#### Option 1: Run with Docker

```bash
docker run -it --gpus all -p 8000:8000 tnlbt-recipe-retrieval bash
```

This command will:

- Start the container with GPU access (`--gpus all`)
- Expose port 8000 (as defined in the Dockerfile) for any web applications

#### Option 2: Run with Docker Compose

If you prefer using Docker Compose, simply run:

```bash
docker-compose up
```

This will start the service using the configuration defined in the `compose.yml` file, which also exposes port 8000.

```
docker build -t soccer-player2vec .
```

```
docker run -it --name soccer-player2vec --gpus all -v "$PWD":/workspace soccer-player2vec /bin/bash
```

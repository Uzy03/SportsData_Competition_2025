```
docker build -t soccer-player2vec .
```

```
docker run -it --rm --gpus all -v "$PWD":/workspace soccer-player2vec /bin/bash
```

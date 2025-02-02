## Install
Build docker file:
```
docker build -t rob521 .
```

To run the container:
```
docker compose run --rm rob521
```

## Testing
Go to the `nodes` directory of the relevant package (for example, `snowball/lab2/nodes`) and run:
```
PYTHONPATH=$PWD pytest
```
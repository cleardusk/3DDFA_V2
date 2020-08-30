## How to fun FaceBoxes

### Build the cpu version of NMS
```shell script
cd utils
python3 build.py build_ext --inplace
```

or just run

```shell script
sh ./build_cpu_nms.sh
```

### Run the demo of face detection
```shell script
python3 FaceBoxes.py
```
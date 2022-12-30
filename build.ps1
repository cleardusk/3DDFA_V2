cd FaceBoxes/utils
./venv/Scripts/python build.py build_ext --inplace
cd ../..

cd Sim3DR
./venv/Scripts/python setup.py build_ext --inplace
cd ..

cd utils/asset
# gcc -shared -Wall -O3 render.c -o render.so -fPIC
cd ../..
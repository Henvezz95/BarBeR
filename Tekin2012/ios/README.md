# Compilação

Na raíz do projeto, crie uma pasta de build:

```
mkdir build && cd build
```
Crie o projeto usando o CMAKE:

```

cmake ../ -GXcode -DCMAKE_TOOLCHAIN_FILE=../ios/ios.toolchain.cmake -DIOS_DEPLOYMENT_TARGET=8.0

```

Para compilar a lib, execute:

```
xcodebuild -target ALL_BUILD -configuration Release -xcconfig ../ios/build.xcconfig
```


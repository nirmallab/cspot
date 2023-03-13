
# remove existing container
docker rmi --force gatorpy
docker system prune --all --force --volumes

# build a new image
docker build -t gatorpy .

# login to push
docker login
docker tag gatorpy nirmallab/gatorpy:gatorpy

# push to docker hub
docker push nirmallab/gatorpy:gatorpy

# pull from docker hub
docker pull nirmallab/gatorpy:gatorpy


# interactive
docker run -it --mount type=bind,source=/Users/aj/Desktop/gatorExampleData,target=/data --name gatorpy gatorpy /bin/bash

# full script
export projectDir="/Users/aj/Desktop/gatorExampleData"
docker run -it --mount type=bind,source=$projectDir,target=/$projectDir \
                gatorpy \
                python /app/generateThumbnails.py \
                --spatialTablePath $projectDir/quantification/exampleSpatialTable.csv \
                --imagePath $projectDir/image/exampleImage.tif \
                --markerChannelMapPath $projectDir/markers.csv \
                --markers ECAD CD3D \
                --maxThumbnails 100 \
                --projectDir $projectDir






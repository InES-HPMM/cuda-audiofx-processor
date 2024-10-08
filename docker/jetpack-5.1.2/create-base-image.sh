# Download the JetPack SDK Docker Image here  https://developer.nvidia.com/downloads/sdkmanager/secure/clients/sdkmanager-2.1.0.11669/sdkmanager-2.1.0.11669-ubuntu_20.04_docker.tar.gz
docker load -i ~/Downloads/sdkmanager-2.1.0.11669-Ubuntu_20.04_docker.tar.gz
sudo docker run -it --name jetpack-sdk-host-5.1.2 --network host sdkmanager:2.1.0.11669-Ubuntu_20.04 --exit-on-finish --collect-usage-data disable --user themuron@hotmail.com --password 1sv7WhcTpvnA --cli --action install --login-type devzone --product Jetson --target-os Linux --version 5.1.2 --show-all-versions --host --license accept
docker commit jetpack-sdk-host-5.1.2 jetpack-sdk-host:5.1.2

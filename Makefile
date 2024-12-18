ZED_SDK_INSTALLER := ZED_SDK_Tegra_L4T35.3_v4.1.0.zstd.run

.PHONY: reformat
reformat:
	black */*.py

.PHONY: test
test:
	cd test && pytest test*.py

.PHONY: install
install:
	python3 -m pip install .

.PHONY: whl
whl:
	apt install -y python3.8-venv
	python3 -m venv venv
	. venv/bin/activate
	python3 -m pip install build
	python3 -m build

.PHONY: zed
zed:
	apt install zstd
	wget --quiet -O $${ZED_SDK_INSTALLER} https://download.stereolabs.com/zedsdk/4.1/l4t35.2/jetsons
	chmod +x $${ZED_SDK_INSTALLER} && ./$${ZED_SDK_INSTALLER} -- silent

# if bison folder exists, skip building.
if [ ! -d bison ]
then
	mkdir bison

	cd bison

	curl https://ftp.gnu.org/gnu/bison/bison-3.8.2.tar.xz -o bison.tar.xz
	tar -xvJf bison.tar.xz
	rm bison.tar.xz

	cd bison-3.8.2
	./configure

	sudo make install -j$(nproc)
fi

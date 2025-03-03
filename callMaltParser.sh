cd ./MaltParser
echo $1
java -Xmx1024m -jar maltparser-1.9.2.jar -c espmalt-1.0 -i $1 -o ../maltParserOut.conll -m parse 
cd ..
echo output saved in ${PWD}"/maltParserOut.conll"
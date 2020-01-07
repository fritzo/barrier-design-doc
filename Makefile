all: main.pdf

main.pdf: FORCE
	pdflatex main
	bibtex main
	pdflatex main

probprog-2020-instructions.pdf: FORCE
	pdflatex probprog-2020-instructions

clean: FORCE
	git clean -dFX

FORCE:

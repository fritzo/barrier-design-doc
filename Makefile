all: probprog-2020-instructions.pdf

probprog-2020-instructions.pdf: FORCE
	pdflatex probprog-2020-instructions

clean: FORCE
	git clean -dFX

FORCE:

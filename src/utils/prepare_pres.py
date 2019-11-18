import argparse
from bs4 import BeautifulSoup


def prepare_jupyter_pres(ifilename, ofilename):
    """
    Remove input cells and 'Out[...]' texts from a html jupyter notebook presentation.

    Parameters
    ==========
    ifilename : str
        Path to input file (html file).
    ofilename : str
        Path to output file (html file).

    Returns
    =======
    Nothing, it writes output into a new html file.
    """

    # Read html file.
    with open(ifilename,'r') as ifile: html=ifile.read()

    # Prepare soup.
    soup = BeautifulSoup(html, "html.parser")

    # Remove input cells.
    for div in soup.find_all("div", {'class':'input'}): div.decompose()

    # Remove 'Out[...]' texts.
    for a in soup.findAll('div',{'class':'prompt output_prompt'}): a.string.replace_with('')

    # Export new html file.
    with open(ofilename, "w") as file: file.write(str(soup))

# Execute main function.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Prepare a Jupyter html presentation (i.e. remove input cells and Out[...] texts).')
    parser.add_argument("-i", "--ifilename", help="Path to input file.", type=str, required=True)
    parser.add_argument("-o", "--ofilename", help="Destination file.", type=str, required=True)
    args = parser.parse_args()
    prepare_jupyter_pres(args.ifilename, args.ofilename)

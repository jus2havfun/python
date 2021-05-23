import asn1tools
import argparse

def load_args():
    parser = argparse.ArgumentParser(description='To parse Sasktel ASN1 files')
    parser.add_argument("--data_file", type=str, default="")
    return parser.parse_args()

args = load_args()
if (args.data_file is not None and len(args.data_file) > 0) :
    file = open(args.data_file, 'rb')
    data = file.read()
    parser = asn1tools.compile_files('Sasktel_TAP3.asn1')
    decoded = parser.decode('DataInterChange', data)
    print (decoded)

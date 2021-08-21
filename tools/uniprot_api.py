import requests, sys
import os
import json

def getGeneFromApi(
        uniprot_id, 
        write=False,
        outdir='../data/api_out'
    ):
    params = {
        'accession': uniprot_id,
    }
    requestURL = "https://www.ebi.ac.uk/proteins/api/coordinates"
    r = requests.get(
        requestURL, 
        headers={"Accept" : "application/json"},
        params=params
    )
    if not r.ok:
        r.raise_for_status()
        sys.exit()
    responseBody = r.json()
    if write:
        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)
        out = '{}/{}.json'.format(outdir, uniprot_id)
        with open(out, 'w') as f:
            json.dump(responseBody, f)
        print('written out', out)
    return responseBody

def sequence(uniprot_id):
    gene_data = getGeneFromApi(uniprot_id)
    if not len(gene_data):
        print('no sequence found for gene {}'.format(uniprot_id))
        return ''
    return gene_data[0]['sequence']
    
if __name__ == '__main__':
    gene_id = 'Q9Y5B0'
    print(gene_id, sequence(gene_id))
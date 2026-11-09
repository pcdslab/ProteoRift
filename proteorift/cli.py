"""CLI interface for ProteoRift"""

import argparse
import logging
from .search import ProteoRiftSearch


def main():
    """Command-line interface for ProteoRift"""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="ProteoRift: Peptide database search with machine learning"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Run database search')
    search_parser.add_argument('--mgf-dir', required=True, help='Directory with MGF files')
    search_parser.add_argument('--peptide-db', required=True, help='Peptide database directory')
    search_parser.add_argument('--output-dir', default='./output', help='Output directory')
    search_parser.add_argument('--tolerance', type=float, default=7, help='Precursor tolerance')
    search_parser.add_argument('--tolerance-type', default='ppm', choices=['ppm', 'Da'])
    search_parser.add_argument('--charge', type=int, default=4, help='Charge state')
    
    # Sample search command
    sample_parser = subparsers.add_parser('search-sample', help='Run search with sample data')
    sample_parser.add_argument('--output-dir', default='./output', help='Output directory')
    sample_parser.add_argument('--no-cache', action='store_true', help='Reprocess from raw')
    
    # Download models command
    download_parser = subparsers.add_parser('download-models', help='Download models from HF Hub')
    download_parser.add_argument('--cache-dir', help='Custom cache directory')
    
    args = parser.parse_args()
    
    if args.command == 'search':
        searcher = ProteoRiftSearch(
            precursor_tolerance=args.tolerance,
            precursor_tolerance_type=args.tolerance_type,
            charge=args.charge
        )
        results = searcher.search(
            mgf_dir=args.mgf_dir,
            peptide_db=args.peptide_db,
            output_dir=args.output_dir
        )
        print(f"Results: {results}")
        
    elif args.command == 'search-sample':
        searcher = ProteoRiftSearch()
        results = searcher.search_with_sample_data(
            output_dir=args.output_dir,
            use_preprocessed=not args.no_cache
        )
        print(f"Results: {results}")
        
    elif args.command == 'download-models':
        from .models import download_models
        download_models(args.cache_dir)
        print("Models downloaded successfully!")
        
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

"""
Vectorless RAG Demo Application
Assortment Planning, Supply Chain SOPs, and Financial Report Analysis

This application demonstrates how to use vectorless RAG for precise
information extraction from structured documents without vector embeddings.
"""

import json
import os
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.tree import Tree

from pageindex import DocumentIndex, PageIndexNavigator, QueryResult

# Load environment variables
load_dotenv()

console = Console()


def load_sec_filing() -> DocumentIndex:
    """Load the sample SEC 10-K filing."""
    sample_path = Path(__file__).parent / "sample_data" / "sec_10k_sample.json"
    
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")
    
    with open(sample_path) as f:
        data = json.load(f)
    
    return DocumentIndex.from_sec_filing(data)


def load_supply_chain_sop() -> DocumentIndex:
    """Load the sample Supply Chain / Assortment Planning SOP."""
    sample_path = Path(__file__).parent / "sample_data" / "assortment_planning_guide.json"
    
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")
    
    with open(sample_path) as f:
        data = json.load(f)
    
    return DocumentIndex.from_supply_chain_sop(data)


def display_document_structure(index: DocumentIndex, max_depth: int = 3):
    """Display the document's hierarchical structure."""
    doc_type = index.metadata.get('document_type', 'Document')
    
    if 'SEC' in doc_type or 'Filing' in doc_type:
        title = f"[FILING] {index.metadata.get('company', '')} {index.metadata.get('filing_type', '')}"
    else:
        title = f"[SOP] {index.metadata.get('title', index.document_id)}"
    
    tree = Tree(title)
    
    def add_children(parent_tree, node, depth=0):
        if depth >= max_depth:
            if node.children:
                parent_tree.add("[dim]...[/dim]")
            return
        for child_id, child in node.children.items():
            icon = "[+]" if child.children else "[-]"
            child_tree = parent_tree.add(f"{icon} {child_id}: {child.title}")
            add_children(child_tree, child, depth + 1)
    
    add_children(tree, index.root)
    console.print(tree)


def display_result(result: QueryResult, show_trace: bool = True):
    """Display query results in a formatted way."""
    # Answer panel
    console.print(Panel(
        Markdown(result.answer),
        title="Answer",
        border_style="green"
    ))
    
    # Sources table
    if result.sources:
        table = Table(title="Sources")
        table.add_column("Section", style="cyan")
        for source in result.sources:
            table.add_row(source)
        console.print(table)
    
    # Navigation path
    console.print(f"\n[dim]Navigation Path: {' → '.join(result.navigation_path)}[/dim]")
    
    # Navigation trace (optionally shown)
    if show_trace:
        console.print(Panel(
            "\n".join(result.reasoning_trace),
            title="Navigation Trace",
            border_style="dim"
        ))
    
    # Confidence
    confidence_color = "green" if result.confidence > 0.7 else "yellow" if result.confidence > 0.4 else "red"
    console.print(f"\n[{confidence_color}]Confidence: {result.confidence:.0%}[/{confidence_color}]")


def interactive_mode(navigator: PageIndexNavigator, index: DocumentIndex):
    """Run in interactive query mode."""
    doc_type = index.metadata.get('document_type', 'document')
    console.print(f"\n[bold cyan]Interactive Query Mode - {doc_type}[/bold cyan]")
    console.print("Type your questions about the document. Type 'quit' to exit.\n")
    
    while True:
        query = console.input("[bold green]Query:[/bold green] ")
        
        if query.lower() in ('quit', 'exit', 'q'):
            break
        
        if not query.strip():
            continue
        
        console.print("\n[dim]Navigating document structure...[/dim]\n")
        
        with console.status("[bold blue]Analyzing document..."):
            result = navigator.query(index, query)
        
        display_result(result)
        console.print("\n" + "="*60 + "\n")


def run_sec_demo(navigator: PageIndexNavigator, index: DocumentIndex):
    """Run SEC 10-K filing demonstration queries."""
    demo_queries = [
        "What is ACME Corporation's total revenue for FY2025 and how did it grow compared to previous year?",
        "What are the main cybersecurity risks and what is the cyber insurance coverage?",
        "What is the company's debt structure and are all debt covenants being met?",
        "What is the CEO's total compensation and what is the CEO pay ratio?",
    ]
    
    console.print(Panel(
        "This demo shows how PageIndex navigates complex SEC filings\n"
        "to extract precise information using LLM reasoning instead of vector search.",
        title="Vectorless RAG Demo - SEC 10-K Analysis",
        border_style="blue"
    ))
    
    console.print("\n[bold]Document Structure:[/bold]\n")
    display_document_structure(index)
    console.print("\n")
    
    _run_queries(navigator, index, demo_queries)


def run_supply_chain_demo(navigator: PageIndexNavigator, index: DocumentIndex):
    """Run Supply Chain / Assortment Planning demonstration queries."""
    demo_queries = [
        "What is the markdown policy for dairy products?",
        "What are the reorder points for high-velocity items?",
        "What is the complete process for discontinuing items from assortment?",
        "What are the vendor performance metrics and targets?",
        "What are the financial requirements for new product selection?",
    ]
    
    console.print(Panel(
        "This demo shows how PageIndex navigates supply chain SOPs and\n"
        "assortment planning guides to extract precise procedures and policies.\n\n"
        "[yellow]Use Case: Supply Chain & Retail Operations[/yellow]\n"
        "Querying markdown policies, inventory rules, or vendor requirements\n"
        "where precise, non-hallucinated answers are mandatory.",
        title="Vectorless RAG Demo - Assortment Planning SOP",
        border_style="blue"
    ))
    
    console.print("\n[bold]Document Structure:[/bold]\n")
    display_document_structure(index)
    console.print("\n")
    
    _run_queries(navigator, index, demo_queries)


def _run_queries(navigator: PageIndexNavigator, index: DocumentIndex, queries: list):
    """Execute a list of demo queries."""
    for i, query in enumerate(queries, 1):
        console.print(Panel(query, title=f"Query {i}", border_style="yellow"))
        
        with console.status("[bold blue]Navigating document tree..."):
            result = navigator.query(index, query)
        
        display_result(result)
        console.print("\n" + "="*80 + "\n")
        
        # Pause between queries in demo mode
        if i < len(queries):
            console.input("[dim]Press Enter for next query...[/dim]")


def main():
    """Main entry point."""
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY environment variable not set.[/red]")
        console.print("Please set it in your .env file or environment.")
        return
    
    # Initialize components
    console.print("[dim]Initializing PageIndex...[/dim]")
    client = OpenAI(api_key=api_key)
    navigator = PageIndexNavigator(client)
    
    # Document type selection
    console.print("\n[bold]Select Document Type:[/bold]")
    console.print("1. Supply Chain SOP (Assortment Planning)")
    console.print("2. SEC 10-K Filing (Financial Report)")
    
    doc_choice = console.input("\nDocument (1/2) [default: 1]: ").strip() or "1"
    
    try:
        if doc_choice == "2":
            console.print("[dim]Loading SEC 10-K filing...[/dim]")
            index = load_sec_filing()
            console.print(f"[green]Loaded {index.metadata['company']} {index.metadata['filing_type']}[/green]\n")
        else:
            console.print("[dim]Loading Supply Chain SOP...[/dim]")
            index = load_supply_chain_sop()
            console.print(f"[green]Loaded {index.metadata['title']}[/green]\n")
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        return
    
    # Mode selection
    console.print("[bold]Select Mode:[/bold]")
    console.print("1. Run demo queries")
    console.print("2. Interactive mode")
    
    mode_choice = console.input("\nMode (1/2) [default: 1]: ").strip() or "1"
    
    if mode_choice == "2":
        interactive_mode(navigator, index)
    else:
        if doc_choice == "2":
            run_sec_demo(navigator, index)
        else:
            run_supply_chain_demo(navigator, index)
    
    console.print("\n[bold green]Thank you for trying Vectorless RAG![/bold green]")


if __name__ == "__main__":
    main()

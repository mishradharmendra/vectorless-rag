"""
PageIndex: Vectorless RAG Implementation
A hierarchical, tree-based retrieval system for complex document navigation.

This module implements the core PageIndex algorithm that replaces traditional
vector databases with LLM-driven reasoning for precise information retrieval.
"""

import json
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from openai import OpenAI
from pydantic import BaseModel


class NavigationAction(str, Enum):
    """Actions the navigator can take when traversing the document tree."""
    DESCEND = "descend"      # Go deeper into a subsection
    EXTRACT = "extract"       # Extract information from current section
    BACKTRACK = "backtrack"   # Go back up the tree
    COMPLETE = "complete"     # Finished gathering information


class NavigationDecision(BaseModel):
    """LLM's decision on how to navigate the document tree."""
    action: NavigationAction
    target_section: Optional[str] = None
    reasoning: str
    extracted_info: Optional[str] = None
    confidence: float = 0.0


class QueryResult(BaseModel):
    """Final result of a document query."""
    answer: str
    sources: list[str]
    confidence: float
    navigation_path: list[str]
    reasoning_trace: list[str]


@dataclass
class DocumentNode:
    """Represents a node in the document tree structure."""
    id: str
    title: str
    content: str
    level: int
    parent: Optional['DocumentNode'] = None
    children: dict[str, 'DocumentNode'] = field(default_factory=dict)
    
    def get_table_of_contents(self, max_depth: int = 2) -> str:
        """Generate a table of contents view from this node."""
        lines = []
        self._build_toc(lines, 0, max_depth)
        return "\n".join(lines)
    
    def _build_toc(self, lines: list[str], current_depth: int, max_depth: int):
        indent = "  " * current_depth
        lines.append(f"{indent}- {self.id}: {self.title}")
        if current_depth < max_depth:
            for child in self.children.values():
                child._build_toc(lines, current_depth + 1, max_depth)
    
    def get_content_preview(self, max_chars: int = 500) -> str:
        """Get a preview of the section content."""
        if len(self.content) <= max_chars:
            return self.content
        return self.content[:max_chars] + "..."


@dataclass
class DocumentIndex:
    """The main PageIndex structure representing a document."""
    document_id: str
    metadata: dict
    root: DocumentNode
    nodes_by_id: dict[str, DocumentNode] = field(default_factory=dict)
    
    @classmethod
    def from_sec_filing(cls, data: dict) -> 'DocumentIndex':
        """Build a DocumentIndex from SEC filing JSON structure."""
        root = DocumentNode(
            id="root",
            title=f"{data['company']} {data['filing_type']} FY{data['fiscal_year']}",
            content="",
            level=0
        )
        
        index = cls(
            document_id=data['document_id'],
            metadata={
                'company': data['company'],
                'filing_type': data['filing_type'],
                'fiscal_year': data['fiscal_year'],
                'document_type': 'SEC Filing'
            },
            root=root
        )
        index.nodes_by_id['root'] = root
        
        # Build the section tree
        for part_id, part_data in data.get('sections', {}).items():
            index._add_section(root, part_id, part_data, 1)
        
        # Add footnotes as a separate branch
        if 'footnotes' in data:
            footnotes_node = DocumentNode(
                id="Footnotes",
                title="Financial Statement Footnotes",
                content="",
                level=1,
                parent=root
            )
            root.children['Footnotes'] = footnotes_node
            index.nodes_by_id['Footnotes'] = footnotes_node
            
            for note_id, note_data in data['footnotes'].items():
                note_node = DocumentNode(
                    id=note_id,
                    title=note_data['title'],
                    content=note_data['content'],
                    level=2,
                    parent=footnotes_node
                )
                footnotes_node.children[note_id] = note_node
                index.nodes_by_id[note_id] = note_node
        
        return index

    @classmethod
    def from_supply_chain_sop(cls, data: dict) -> 'DocumentIndex':
        """Build a DocumentIndex from Supply Chain/Assortment Planning SOP JSON structure."""
        root = DocumentNode(
            id="root",
            title=data.get('title', 'Supply Chain SOP'),
            content=f"Document ID: {data.get('document_id', 'N/A')}\n"
                    f"Version: {data.get('version', 'N/A')}\n"
                    f"Effective Date: {data.get('effective_date', 'N/A')}\n"
                    f"Classification: {data.get('classification', 'N/A')}",
            level=0
        )
        
        index = cls(
            document_id=data.get('document_id', 'unknown'),
            metadata={
                'title': data.get('title'),
                'document_type': data.get('document_type', 'Standard Operating Procedure'),
                'version': data.get('version'),
                'effective_date': data.get('effective_date'),
                'classification': data.get('classification')
            },
            root=root
        )
        index.nodes_by_id['root'] = root
        
        # Build the main section tree
        for section_id, section_data in data.get('sections', {}).items():
            index._add_section(root, section_id, section_data, 1)
        
        # Add appendices as a separate branch
        if 'appendices' in data:
            appendices_node = DocumentNode(
                id="Appendices",
                title="Appendices",
                content="",
                level=1,
                parent=root
            )
            root.children['Appendices'] = appendices_node
            index.nodes_by_id['Appendices'] = appendices_node
            
            for app_id, app_data in data['appendices'].items():
                app_node = DocumentNode(
                    id=app_id,
                    title=app_data['title'],
                    content=app_data['content'],
                    level=2,
                    parent=appendices_node
                )
                appendices_node.children[app_id] = app_node
                index.nodes_by_id[app_id] = app_node
        
        return index
    
    def _add_section(self, parent: DocumentNode, section_id: str, 
                     section_data: dict, level: int):
        """Recursively add sections to the tree."""
        node = DocumentNode(
            id=section_id,
            title=section_data.get('title', section_id),
            content=section_data.get('content', ''),
            level=level,
            parent=parent
        )
        parent.children[section_id] = node
        self.nodes_by_id[section_id] = node
        
        # Process subsections
        for sub_id, sub_data in section_data.get('subsections', {}).items():
            self._add_section(node, sub_id, sub_data, level + 1)


class PageIndexNavigator:
    """
    LLM-powered navigator for traversing and extracting from PageIndex.
    
    Unlike vector similarity search, this navigator uses LLM reasoning
    to make intelligent navigation decisions based on the query semantics
    and document structure.
    """
    
    NAVIGATOR_SYSTEM_PROMPT = """You are a document navigation expert analyzing business documents, supply chain SOPs, retail operations manuals, regulatory filings, and assortment planning guides.
Your task is to navigate a hierarchical document structure to find precise information.

NAVIGATION RULES:
1. Start from the table of contents and identify the most relevant section
2. DESCEND into sections that likely contain the answer
3. EXTRACT when you find the exact information needed
4. BACKTRACK if you went to the wrong section
5. COMPLETE when you have gathered enough information to answer

IMPORTANT:
- Be precise - extract exact numbers, specifications, procedures, policies, and requirements
- Follow document hierarchy logically (Section -> Subsection -> Paragraph)
- If information spans multiple sections, navigate to each
- For business-critical information, verify you have the complete requirement
- For cross-references (e.g., "see Section 2.3"), navigate to extract that section too
- Confidence should reflect how well the extracted info answers the query"""

    NAVIGATION_PROMPT = """Current Location: {current_location}
Query: {query}

Available Sections:
{available_sections}

Current Section Content:
{current_content}

Previously Extracted Information:
{extracted_info}

Navigation Path So Far: {nav_path}

Decide your next action. Return JSON with:
- action: "descend" (go into a subsection), "extract" (capture info here), "backtrack" (go up), or "complete" (done)
- target_section: section ID to descend into (if action is descend)
- reasoning: why you're taking this action
- extracted_info: information extracted (if action is extract)
- confidence: 0.0-1.0 how confident you are this helps answer the query"""

    def __init__(self, client: OpenAI, model: str = "gpt-4o"):
        self.client = client
        self.model = model
    
    def query(self, index: DocumentIndex, query: str, 
              max_steps: int = 15) -> QueryResult:
        """
        Navigate the document index to answer a query.
        
        This implements the core vectorless RAG algorithm:
        1. Present the document structure (not embeddings) to the LLM
        2. Let the LLM decide which section to explore based on reasoning
        3. Traverse the tree, extracting relevant information
        4. Synthesize a final answer from extracted pieces
        """
        current_node = index.root
        navigation_path = ["root"]
        reasoning_trace = []
        extracted_pieces = []
        sources = []
        
        for step in range(max_steps):
            # Build context for the LLM
            available_sections = self._format_available_sections(current_node)
            
            # Get navigation decision from LLM
            decision = self._get_navigation_decision(
                query=query,
                current_node=current_node,
                available_sections=available_sections,
                extracted_so_far=extracted_pieces,
                nav_path=navigation_path,
                doc_type=index.metadata.get('document_type', 'Document')
            )
            
            reasoning_trace.append(f"Step {step + 1}: {decision.action.value} - {decision.reasoning}")
            
            if decision.action == NavigationAction.DESCEND:
                if decision.target_section and decision.target_section in current_node.children:
                    current_node = current_node.children[decision.target_section]
                    navigation_path.append(decision.target_section)
                else:
                    reasoning_trace.append(f"  -> Invalid section '{decision.target_section}', staying at current location")
            
            elif decision.action == NavigationAction.EXTRACT:
                if decision.extracted_info:
                    extracted_pieces.append(decision.extracted_info)
                    sources.append(f"{current_node.id}: {current_node.title}")
            
            elif decision.action == NavigationAction.BACKTRACK:
                if current_node.parent:
                    current_node = current_node.parent
                    navigation_path.append(f"[up]{current_node.id}")
            
            elif decision.action == NavigationAction.COMPLETE:
                break
        
        # Synthesize final answer
        final_answer = self._synthesize_answer(
            query, 
            extracted_pieces, 
            sources,
            index.metadata.get('document_type', 'Document')
        )
        
        return QueryResult(
            answer=final_answer,
            sources=sources,
            confidence=decision.confidence if extracted_pieces else 0.0,
            navigation_path=navigation_path,
            reasoning_trace=reasoning_trace
        )
    
    def _format_available_sections(self, node: DocumentNode) -> str:
        """Format child sections for LLM display."""
        if not node.children:
            return "(No subsections - this is a leaf section)"
        
        lines = []
        for child_id, child in node.children.items():
            preview = child.get_content_preview(200)
            has_children = "[+]" if child.children else "[-]"
            lines.append(f"{has_children} [{child_id}] {child.title}")
            if preview:
                lines.append(f"   Preview: {preview[:150]}...")
        return "\n".join(lines)
    
    def _get_navigation_decision(self, query: str, current_node: DocumentNode,
                                  available_sections: str, extracted_so_far: list[str],
                                  nav_path: list[str], doc_type: str = "Document") -> NavigationDecision:
        """Ask the LLM to decide the next navigation action."""
        prompt = self.NAVIGATION_PROMPT.format(
            current_location=f"{current_node.id}: {current_node.title}",
            query=query,
            available_sections=available_sections,
            current_content=current_node.get_content_preview(800) or "(No direct content)",
            extracted_info="\n".join(extracted_so_far) if extracted_so_far else "(None yet)",
            nav_path=" -> ".join(nav_path)
        )
        
        system_prompt = self.NAVIGATOR_SYSTEM_PROMPT
        if "SOP" in doc_type or "Procedure" in doc_type or "Planning" in doc_type or "Supply" in doc_type:
            system_prompt += "\n\nThis is a BUSINESS/OPERATIONS document. Pay special attention to:\n- Specific procedure steps and their order\n- Policy requirements and approval thresholds\n- Markdown tiers and timing rules\n- Numerical limits, percentages, and specifications\n- Cross-references to other sections or procedures"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        return NavigationDecision(
            action=NavigationAction(result.get('action', 'complete')),
            target_section=result.get('target_section'),
            reasoning=result.get('reasoning', ''),
            extracted_info=result.get('extracted_info'),
            confidence=float(result.get('confidence', 0.5))
        )
    
    def _synthesize_answer(self, query: str, extracted_pieces: list[str],
                           sources: list[str], doc_type: str = "Document") -> str:
        """Synthesize a final answer from extracted information."""
        if not extracted_pieces:
            return "Unable to find relevant information in the document."
        
        synthesis_prompt = f"""Based on the following extracted information from a {doc_type}, 
provide a precise, well-structured answer to the query.

Query: {query}

Extracted Information:
{chr(10).join(f'- {piece}' for piece in extracted_pieces)}

Sources: {', '.join(sources)}

Provide a clear, factual answer using only the extracted information. 
Include specific numbers, percentages, procedures, and requirements where available.
For business-critical information, ensure completeness and accuracy."""

        system_content = "You are a business documentation specialist providing precise answers."
        if "SOP" in doc_type or "Procedure" in doc_type or "Planning" in doc_type or "Supply" in doc_type:
            system_content += " For business procedures, ensure all steps, thresholds, and policy requirements are clearly stated. Do not omit critical details."
        elif "SEC" in doc_type or "Filing" in doc_type:
            system_content = "You are a financial analyst providing precise answers based on SEC filings."
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": synthesis_prompt}
            ],
            temperature=0.1
        )
        
        return response.choices[0].message.content

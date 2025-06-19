import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as d3 from 'd3';

const GraphVisualization = ({ data, onNodeClick, selectedNodeId, isSidePanelOpen = false }) => {
  const svgRef = useRef();
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [transform, setTransform] = useState(d3.zoomIdentity);

  // Update dimensions on resize
  useEffect(() => {
    const updateDimensions = () => {
      if (svgRef.current && svgRef.current.parentElement) {
        const { width, height } = svgRef.current.parentElement.getBoundingClientRect();
        setDimensions({ width, height });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Initialize and update the graph
  useEffect(() => {
    if (!data?.nodes?.length || !dimensions.width) return;

    const { width, height } = dimensions;
    const svg = d3.select(svgRef.current);
    
    // Clear previous content
    svg.selectAll('*').remove();

    // Create main group for zoom/pan
    const g = svg.append('g');

    // Create zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        setTransform(event.transform);
        g.attr('transform', event.transform);
      });

    svg.call(zoom);

    // Create force simulation
    const simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.links).id(d => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30));

    // Create links
    const link = g.append('g')
      .selectAll('line')
      .data(data.links)
      .enter()
      .append('line')
      .attr('class', 'link')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6);

    // Create nodes
    const node = g.append('g')
      .selectAll('.node')
      .data(data.nodes)
      .enter()
      .append('g')
      .attr('class', 'node')
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended)
      );

    // Add circles to nodes
    node.append('circle')
      .attr('r', 10)
      .attr('fill', d => d.color || '#1f77b4')
      .attr('stroke', d => d.id === selectedNodeId ? '#ff5722' : '#fff')
      .attr('stroke-width', d => d.id === selectedNodeId ? 3 : 2);

    // Add labels
    node.append('text')
      .attr('dy', -15)
      .text(d => d.id)
      .attr('class', 'label')
      .style('pointer-events', 'none')
      .style('font-size', '12px')
      .style('text-anchor', 'middle');

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    // Drag functions
    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    // Node click handler
    node.on('click', (event, d) => {
      event.stopPropagation();
      if (onNodeClick) onNodeClick(event, d);
      
      // Center on clicked node
      const [x, y] = [d.x, d.y];
      const scale = 1.5;
      const newTransform = d3.zoomIdentity
        .translate(width / 2 - x * scale, height / 2 - y * scale)
        .scale(scale);
      
      svg.transition()
        .duration(750)
        .call(zoom.transform, newTransform);
    });

    // Background click handler
    svg.on('click', (event) => {
      if (event.target === svg.node() && onNodeClick) {
        onNodeClick(event, null);
      }
    });

    // Cleanup
    return () => {
      simulation.stop();
      svg.on('.zoom', null);
    };
  }, [data, dimensions, onNodeClick, selectedNodeId]);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <svg 
        ref={svgRef}
        width="100%"
        height="100%"
        style={{
          backgroundColor: '#f8f9fa',
          borderRadius: '4px',
          minHeight: '500px',
        }}
      />
      
      <style jsx global>{`
        .node circle {
          cursor: pointer;
          transition: r 0.2s, stroke-width 0.2s;
        }
        .node circle:hover {
          stroke-width: 3px !important;
          filter: drop-shadow(0 0 4px rgba(0, 0, 0, 0.3));
        }
        .link {
          stroke: #999;
          stroke-opacity: 0.6;
        }
        .label {
          font-size: 12px;
          font-family: sans-serif;
          font-weight: bold;
          fill: #333;
          text-shadow: 0 1px 0 #fff, 1px 0 0 #fff, -1px 0 0 #fff, 0 -1px 0 #fff;
        }
      `}</style>
    </div>
  );
};

export default GraphVisualization;

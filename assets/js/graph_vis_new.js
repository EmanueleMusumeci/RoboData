$(document).ready(function() {
    // Mapping of graph container IDs to graph data IDs
    const graphDataMapping = {
        "batch_a_QA01-graph": "batch_a_QA01",
        "batch_a_QA02-graph": "batch_a_QA02",
        "batch_a_QA03-graph": "batch_a_QA03",
        "batch_a_QA04-graph": "batch_a_QA04",
        "batch_a_QA05-graph": "batch_a_QA05",
        "batch_a_QA06-graph": "batch_a_QA06",
        "batch_a_QA07-graph": "batch_a_QA07",
        "batch_b_QB01-graph": "batch_b_QB01",
        "batch_b_QB02-graph": "batch_b_QB02",
        "batch_b_QB03-graph": "batch_b_QB03",
        "batch_d_QD01-graph": "batch_d_QD01",
        "batch_d_QD02-graph": "batch_d_QD02",
        "batch_d_QD03-graph": "batch_d_QD03",
        "batch_d_QD04-graph": "batch_d_QD04",
        "batch_d_QD06-graph": "batch_d_QD06",
        "batch_d_QD07-graph": "batch_d_QD07",
        "batch_d_QD08-graph": "batch_d_QD08"
    };

    const loadedGraphs = {};

    function getRefColor(className) {
        const tempElement = $(`<div class="${className}" style="display: none;"></div>`);
        $('body').append(tempElement);
        const color = tempElement.css('color');
        tempElement.remove();
        return color;
    }

    const refColors = {};
    for (let i = 1; i <= 10; i++) {
        refColors[`ref${i}`] = getRefColor(`ref${i}`);
    }

    $('.collapsible-prompt .button').on('click', function() {
        const content = $(this).next('.prompt-content');
        const graphContainer = content.find('.graph-container');
        const graphId = graphContainer.attr('id');

        if (content.is(':visible')) {
            // Collapsing - handled by index.html script
        } else {
            // Expanding - load graph if not already loaded
            if (graphId && !loadedGraphs[graphId]) {
                const dataId = graphDataMapping[graphId];
                if (dataId) {
                    loadGraphFromData(graphContainer, dataId, graphId);
                } else {
                    console.error("No data mapping found for graph ID:", graphId);
                    graphContainer.html("Knowledge graph data not available.");
                }
            }
        }
    });

    function loadGraphFromData(container, dataId, graphId) {
        console.log("Loading graph:", dataId, "for container:", graphId);
        
        try {
            // Check if graph data is available
            if (!window.loadGraphData) {
                console.error("Graph data loader not available");
                container.html("Graph data loader not available. Please make sure all graph data scripts are loaded.");
                return;
            }

            const graphData = window.loadGraphData(dataId);
            if (!graphData) {
                console.error("No graph data found for:", dataId);
                container.html("Knowledge graph data not found.");
                return;
            }

            console.log("Graph data loaded:", graphData.nodes.length, "nodes,", graphData.edges.length, "edges");

            // Create vis.js datasets
            const nodes = new vis.DataSet(graphData.nodes);
            const edges = new vis.DataSet(graphData.edges);
            
            console.log("Vis.js datasets created");
            
            // Find the row ID for colorization
            const rowId = container.closest('tr').prev('tr').prev('tr').attr('id');
            colorizeGraphElements(nodes, edges, rowId);

            // Create the network
            const data = { nodes: nodes, edges: edges };
            const options = {
                nodes: {
                    shape: 'box',
                    font: {
                        size: 14,
                        face: 'arial'
                    }
                },
                edges: {
                    font: {
                        size: 12,
                        align: 'middle'
                    },
                    arrows: 'to'
                },
                physics: {
                    solver: 'repulsion',
                    repulsion: {
                        nodeDistance: 200
                    }
                },
                interaction: {
                    dragNodes: true,
                    dragView: true,
                    zoomView: true
                }
            };

            // Set background color for the container
            container.css('background-color', '#ffffff');

            console.log("Creating network with container:", container[0]);
            const network = new vis.Network(container[0], data, options);
            
            // Fit the network when stabilization is done
            network.on("stabilizationIterationsDone", function () {
                network.fit();
                console.log("Network stabilized and fitted");
            });
            
            loadedGraphs[graphId] = { network, nodes, edges };
            setupHoverInteractions(rowId, network, nodes, edges);

            console.log("Successfully loaded graph:", dataId);

        } catch (error) {
            console.error("Error loading graph:", error);
            container.html("Error loading knowledge graph: " + error.message);
        }
    }

    function colorizeGraphElements(nodes, edges, rowId) {
        if (!rowId) return;

        const supportSets = $(`#${rowId} .ref, #${rowId} + tr .ref`);
        supportSets.each(function() {
            const refClass = $(this).attr('class').split(' ').find(c => c.startsWith('ref'));
            if (!refClass) return;

            const color = refColors[refClass];
            const supportText = $(this).text();
            const nodeIds = supportText.match(/Q\d+/g);

            if (nodeIds && nodeIds.length > 0) {
                nodeIds.forEach(id => {
                    if (nodes.get(id)) {
                        nodes.update({ 
                            id: id, 
                            color: { background: color, border: color }, 
                            font: { color: 'white' } 
                        });
                    }
                    // Color edges connected to this node
                    const connectedEdges = edges.get({
                        filter: function(edge) {
                            return edge.from === id || edge.to === id;
                        }
                    });
                    connectedEdges.forEach(edge => {
                        edges.update({ id: edge.id, color: color });
                    });
                });
            }
        });
    }

    function setupHoverInteractions(rowId, network, nodes, edges) {
        const originalNodeColors = {};
        nodes.forEach(node => {
            originalNodeColors[node.id] = { 
                background: node.color?.background, 
                border: node.color?.border, 
                font: node.font?.color 
            };
        });

        const originalEdgeColors = {};
        edges.forEach(edge => {
            originalEdgeColors[edge.id] = edge.color;
        });

        const highlightColor = '#FFFF00'; // Yellow

        $(`#${rowId} .ref, #${rowId} + tr .ref`).hover(
            function() { // MOUSE IN
                const supportText = $(this).text();
                const nodeIds = supportText.match(/Q\d+/g) || [];
                
                // Highlight table cell
                $(this).css('background-color', highlightColor);

                // Highlight graph nodes and edges
                nodeIds.forEach(id => {
                    if (nodes.get(id)) {
                        nodes.update({ 
                            id: id, 
                            color: { background: highlightColor, border: 'black' }, 
                            font: { color: 'black' } 
                        });
                    }
                    const connectedEdges = edges.get({
                        filter: edge => edge.from === id || edge.to === id
                    });
                    connectedEdges.forEach(edge => {
                        edges.update({ id: edge.id, color: highlightColor, width: 3 });
                    });
                });
            },
            function() { // MOUSE OUT
                const supportText = $(this).text();
                const nodeIds = supportText.match(/Q\d+/g) || [];

                // Un-highlight table cell
                $(this).css('background-color', '');

                // Restore original graph colors
                nodeIds.forEach(id => {
                    if (nodes.get(id)) {
                        const original = originalNodeColors[id];
                        nodes.update({ 
                            id: id, 
                            color: { background: original.background, border: original.border }, 
                            font: { color: original.font } 
                        });
                    }
                    const connectedEdges = edges.get({
                        filter: edge => edge.from === id || edge.to === id
                    });
                    connectedEdges.forEach(edge => {
                        edges.update({ id: edge.id, color: originalEdgeColors[edge.id], width: 1 });
                    });
                });
            }
        );
    }
});

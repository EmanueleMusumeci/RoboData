import React, { useState, useRef } from 'react';
import { 
  Box, 
  Typography, 
  IconButton, 
  Paper, 
  List, 
  ListItem, 
  ListItemText,
  Link,
  Collapse,
  Tooltip
} from '@mui/material';
import { 
  Close as CloseIcon,
  Link as LinkIcon,
  ArrowDropDown as ArrowDropDownIcon,
  ArrowRight as ArrowRightIcon,
  Build as BuildIcon,
  ChatBubbleOutline as CommentIcon
} from '@mui/icons-material';

const NodeDetailsPanel = ({ node, onClose, position, isOpen, onDragStart, onDrag, onDragEnd }) => {
  const [expandedSections, setExpandedSections] = useState({
    properties: true,
    neighbors: true,
    tools: true
  });
  const panelRef = useRef(null);

  if (!node) return null;

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const handleDragStart = (e) => {
    if (e.target === panelRef.current) {
      onDragStart(e);
    }
  };

  const handleDrag = (e) => {
    if (e.buttons === 1) { // Only drag if left mouse button is pressed
      onDrag(e);
    }
  };

  const handleDragEnd = (e) => {
    onDragEnd(e);
  };

  // Mock data - replace with actual data from your API
  const nodeData = {
    label: node.label || node.id,
    id: node.id,
    description: 'This is a sample description for the node. In a real implementation, this would come from your data source.',
    image: node.image || 'https://via.placeholder.com/200x150?text=No+Image',
    wikidataUrl: `https://www.wikidata.org/wiki/${node.id}`,
    discussionUrl: `https://www.wikidata.org/wiki/Talk:${node.id}`,
    properties: [
      { id: 'P31', label: 'instance of', value: 'human' },
      { id: 'P21', label: 'sex or gender', value: 'male' },
      { id: 'P569', label: 'date of birth', value: '1980-01-01' },
    ],
    neighbors: [
      { id: 'Q2', label: 'Earth', relation: 'instance of' },
      { id: 'Q5', label: 'Human', relation: 'subclass of' },
    ]
  };

  return (
    <Paper
      ref={panelRef}
      elevation={3}
      style={{
        position: 'absolute',
        left: position.x,
        top: position.y,
        width: '300px',
        maxHeight: '80vh',
        overflowY: 'auto',
        zIndex: 1000,
        cursor: isOpen ? 'move' : 'default',
        display: 'flex',
        flexDirection: 'column',
      }}
      draggable={isOpen}
      onMouseDown={handleDragStart}
      onMouseMove={handleDrag}
      onMouseUp={handleDragEnd}
      onMouseLeave={handleDragEnd}
    >
      <Box sx={{ p: 2, bgcolor: 'primary.main', color: 'white', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h6" noWrap>{nodeData.label}</Typography>
        <IconButton size="small" onClick={onClose} sx={{ color: 'white' }}>
          <CloseIcon />
        </IconButton>
      </Box>
      
      <Box sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <img 
            src={nodeData.image} 
            alt={nodeData.label} 
            style={{ 
              width: '100%', 
              maxHeight: '200px', 
              objectFit: 'cover',
              borderRadius: '4px',
              marginBottom: '8px'
            }} 
          />
        </Box>
        
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {nodeData.description}
        </Typography>
        
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <LinkIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />
          <Link href={nodeData.wikidataUrl} target="_blank" rel="noopener">
            View on Wikidata
          </Link>
        </Box>
        
        {/* Properties Section */}
        <Box sx={{ mb: 2 }}>
          <Box 
            sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              cursor: 'pointer',
              '&:hover': { bgcolor: 'action.hover' },
              p: 0.5,
              borderRadius: 1
            }}
            onClick={() => toggleSection('properties')}
          >
            {expandedSections.properties ? <ArrowDropDownIcon /> : <ArrowRightIcon />}
            <Typography variant="subtitle2" sx={{ ml: 1 }}>Properties</Typography>
          </Box>
          <Collapse in={expandedSections.properties}>
            <List dense>
              {nodeData.properties.map((prop, index) => (
                <ListItem key={index} sx={{ py: 0.5, pl: 4 }}>
                  <Tooltip title={prop.id} placement="left">
                    <span style={{ minWidth: '80px' }}>
                      <Link href={`https://www.wikidata.org/wiki/Property:${prop.id}`} target="_blank" rel="noopener" color="inherit">
                        {prop.label}
                      </Link>
                    </span>
                  </Tooltip>
                  <ListItemText 
                    primary={prop.value} 
                    primaryTypographyProps={{ variant: 'body2' }}
                    sx={{ ml: 2 }}
                  />
                </ListItem>
              ))}
            </List>
          </Collapse>
        </Box>
        
        {/* Neighbors Section */}
        <Box sx={{ mb: 2 }}>
          <Box 
            sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              cursor: 'pointer',
              '&:hover': { bgcolor: 'action.hover' },
              p: 0.5,
              borderRadius: 1
            }}
            onClick={() => toggleSection('neighbors')}
          >
            {expandedSections.neighbors ? <ArrowDropDownIcon /> : <ArrowRightIcon />}
            <Typography variant="subtitle2" sx={{ ml: 1 }}>Neighbors</Typography>
          </Box>
          <Collapse in={expandedSections.neighbors}>
            <List dense>
              {nodeData.neighbors.map((neighbor, index) => (
                <ListItem key={index} sx={{ py: 0.5, pl: 4 }}>
                  <ListItemText 
                    primary={
                      <Link href={`#${neighbor.id}`} color="primary">
                        {neighbor.label}
                      </Link>
                    }
                    secondary={neighbor.relation}
                    primaryTypographyProps={{ variant: 'body2' }}
                    secondaryTypographyProps={{ variant: 'caption' }}
                  />
                </ListItem>
              ))}
            </List>
          </Collapse>
        </Box>
        
        {/* Tools and Discussion */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', pt: 1, borderTop: '1px solid', borderColor: 'divider' }}>
          <Tooltip title="Tools (Coming soon)">
            <IconButton size="small" disabled>
              <BuildIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="View discussion on Wikidata">
            <IconButton 
              size="small" 
              href={nodeData.discussionUrl} 
              target="_blank" 
              rel="noopener"
              component="a"
            >
              <CommentIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
    </Paper>
  );
};

export default NodeDetailsPanel;

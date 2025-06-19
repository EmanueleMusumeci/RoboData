import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  List, 
  ListItem, 
  ListItemText,
  Link,
  Collapse,
  Divider,
  Avatar,
  Chip,
  IconButton,
  Tooltip
} from '@mui/material';
import { 
  Link as LinkIcon,
  ArrowDropDown as ArrowDropDownIcon,
  ArrowRight as ArrowRightIcon,
  Build as BuildIcon,
  ChatBubbleOutline as CommentIcon
} from '@mui/icons-material';

const NodeDetailsTab = ({ node }) => {
  const [expandedSections, setExpandedSections] = useState({
    properties: true,
    neighbors: true,
    tools: true
  });

  if (!node) return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center', 
      justifyContent: 'center',
      height: '100%',
      color: 'text.secondary',
      p: 3,
      textAlign: 'center'
    }}>
      <BuildIcon sx={{ fontSize: 48, mb: 2, opacity: 0.3 }} />
      <Typography variant="h6">No node selected</Typography>
      <Typography variant="body2" sx={{ mt: 1 }}>
        Click on a node in the graph to view its details
      </Typography>
    </Box>
  );

  // Mock data - replace with actual data from your API
  const nodeData = {
    label: node.label || node.id,
    id: node.id,
    description: node.description || 'No description available',
    image: node.image || 'https://via.placeholder.com/200x150?text=No+Image',
    wikidataUrl: `https://www.wikidata.org/wiki/${node.id}`,
    discussionUrl: `https://www.wikidata.org/wiki/Talk:${node.id}`,
    properties: node.properties || [
      { id: 'P31', label: 'instance of', value: 'human' },
      { id: 'P21', label: 'sex or gender', value: 'male' },
    ],
    neighbors: node.neighbors || []
  };

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  return (
    <Box sx={{ 
      height: '100%',
      overflowY: 'auto',
      p: 2,
      display: 'flex',
      flexDirection: 'column',
      gap: 2
    }}>
      {/* Header */}
      <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start' }}>
        <Avatar 
          src={nodeData.image} 
          alt={nodeData.label}
          variant="rounded"
          sx={{ width: 80, height: 80, mt: 0.5 }}
        />
        <Box sx={{ flex: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
            <Typography variant="h6" noWrap sx={{ fontWeight: 'bold' }}>
              {nodeData.label}
            </Typography>
            <Chip 
              label={nodeData.id} 
              size="small" 
              variant="outlined"
              component="a"
              href={nodeData.wikidataUrl}
              target="_blank"
              rel="noopener noreferrer"
              clickable
              icon={<LinkIcon fontSize="small" />}
              sx={{ height: 22, '& .MuiChip-label': { px: 1 } }}
            />
          </Box>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            {nodeData.description}
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, mt: 'auto' }}>
            <Tooltip title="Open in Wikidata">
              <IconButton 
                size="small" 
                href={nodeData.wikidataUrl}
                target="_blank"
                rel="noopener noreferrer"
              >
                <LinkIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Discussion">
              <IconButton 
                size="small" 
                href={nodeData.discussionUrl}
                target="_blank"
                rel="noopener noreferrer"
              >
                <CommentIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
      </Box>


      {/* Properties Section */}
      <Box>
        <Box 
          onClick={() => toggleSection('properties')}
          sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            cursor: 'pointer',
            '&:hover': { bgcolor: 'action.hover' },
            p: 0.5,
            borderRadius: 1
          }}
        >
          {expandedSections.properties ? <ArrowDropDownIcon /> : <ArrowRightIcon />}
          <Typography variant="subtitle2" sx={{ ml: 0.5, fontWeight: 'bold' }}>
            Properties
          </Typography>
        </Box>
        <Collapse in={expandedSections.properties}>
          <List dense sx={{ py: 0, pl: 3 }}>
            {nodeData.properties.map((prop, index) => (
              <ListItem key={index} disableGutters sx={{ py: 0.5 }}>
                <ListItemText 
                  primary={
                    <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                      <Chip 
                        label={prop.id} 
                        size="small" 
                        variant="outlined"
                        sx={{ height: 20, '& .MuiChip-label': { px: 0.75 } }}
                      />
                      <Typography variant="body2" component="span">
                        {prop.label}:
                      </Typography>
                      <Typography variant="body2" component="span" sx={{ fontWeight: 'medium' }}>
                        {prop.value}
                      </Typography>
                    </Box>
                  }
                />
              </ListItem>
            ))}
          </List>
        </Collapse>
      </Box>

      {nodeData.neighbors.length > 0 && (
        <Box>
          <Box 
            onClick={() => toggleSection('neighbors')}
            sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              cursor: 'pointer',
              '&:hover': { bgcolor: 'action.hover' },
              p: 0.5,
              borderRadius: 1
            }}
          >
            {expandedSections.neighbors ? <ArrowDropDownIcon /> : <ArrowRightIcon />}
            <Typography variant="subtitle2" sx={{ ml: 0.5, fontWeight: 'bold' }}>
              Connections
            </Typography>
          </Box>
          <Collapse in={expandedSections.neighbors}>
            <List dense sx={{ py: 0, pl: 3 }}>
              {nodeData.neighbors.map((neighbor, index) => (
                <ListItem key={index} disableGutters sx={{ py: 0.5 }}>
                  <ListItemText 
                    primary={
                      <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                        <Typography variant="body2" component="span">
                          {neighbor.relation}:
                        </Typography>
                        <Link 
                          href={`#${neighbor.id}`} 
                          underline="hover"
                          sx={{ cursor: 'pointer' }}
                        >
                          {neighbor.label}
                        </Link>
                      </Box>
                    }
                  />
                </ListItem>
              ))}
            </List>
          </Collapse>
        </Box>
      )}
    </Box>
  );
};

export default NodeDetailsTab;

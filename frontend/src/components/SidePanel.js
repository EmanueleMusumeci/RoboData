import React, { useState, useRef, useEffect } from 'react';
import { 
  Box, 
  IconButton, 
  Tabs, 
  Tab, 
  Tooltip,
  Paper,
  InputBase
} from '@mui/material';
import { 
  Close as CloseIcon,
  Add as AddIcon,
  Search as SearchIcon,
  ChevronRight as ChevronRightIcon,
  ArrowBackIos as ArrowBackIosIcon,
  ArrowForwardIos as ArrowForwardIosIcon
} from '@mui/icons-material';
import NodeDetailsTab from './NodeDetailsTab';

const TabPanel = ({ children, value, index, ...other }) => {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`side-panel-tabpanel-${index}`}
      aria-labelledby={`side-panel-tab-${index}`}
      style={{ height: '100%' }}
      {...other}
    >
      {value === index && (
        <Box sx={{ height: '100%', overflow: 'auto' }}>
          {children}
        </Box>
      )}
    </div>
  );
};

const SidePanel = ({ 
  isOpen, 
  onClose, 
  selectedNode, 
  onNodeSelect,
  nodes = []
}) => {
  const [tabValue, setTabValue] = useState(0);
  const [tabs, setTabs] = useState([
    { id: 1, title: 'New Tab', node: null }
  ]);
  const [searchQuery, setSearchQuery] = useState('');
  const panelRef = useRef(null);

  // Handle tab changes
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  // Add a new tab
  const handleAddTab = () => {
    const newTab = {
      id: Date.now(),
      title: 'New Tab',
      node: null
    };
    setTabs([...tabs, newTab]);
    setTabValue(tabs.length);
  };

  // Update tab when node is selected
  useEffect(() => {
    if (selectedNode) {
      // Always create a new tab when a node is selected
      const newTab = {
        id: Date.now(),
        title: selectedNode.label || selectedNode.id,
        node: selectedNode
      };
      setTabs(prevTabs => {
        const newTabs = [...prevTabs, newTab];
        setTabValue(newTabs.length - 1);
        return newTabs;
      });
    }
  }, [selectedNode]);

  return (
    <Box
      ref={panelRef}
      sx={{
        position: 'absolute',
        right: isOpen ? 0 : '-400px',
        top: 0,
        bottom: 0,
        width: '400px',
        bgcolor: 'background.paper',
        boxShadow: 3,
        transition: 'right 0.3s ease-in-out',
        display: 'flex',
        flexDirection: 'column',
        zIndex: 1200,
        borderLeft: '1px solid',
        borderColor: 'divider',
        '&:hover .collapse-button': {
          opacity: 1
        }
      }}
    >
      {/* Tab bar with navigation arrows and collapse button */}
      <Box 
        sx={{ 
          borderBottom: 1, 
          borderColor: 'divider',
          display: 'flex',
          alignItems: 'center',
          bgcolor: 'background.default',
          minHeight: '48px',
          position: 'relative',
          pr: 6 // Make space for the collapse button
        }}
      >
        {/* Left arrow */}
        <IconButton 
          size="small" 
          disabled={tabValue === 0}
          onClick={() => setTabValue(prev => Math.max(0, prev - 1))}
          sx={{ 
            ml: 0.5,
            my: 'auto',
            visibility: tabs.length > 1 ? 'visible' : 'hidden'
          }}
        >
          <ArrowBackIosIcon fontSize="small" />
        </IconButton>

        <Box sx={{ 
          display: 'flex', 
          flex: 1, 
          overflow: 'hidden',
          justifyContent: 'center' 
        }}>
          <Tabs
            value={Math.min(tabValue, tabs.length - 1)}
            onChange={handleTabChange}
            variant="scrollable"
            scrollButtons={false}
            sx={{
              minHeight: '48px',
              '& .MuiTabs-scroller': {
                display: 'flex',
                alignItems: 'center'
              },
              '& .MuiTabs-flexContainer': {
                justifyContent: 'center'
              }
            }}
          >
            {tabs.map((tab, index) => (
              <Tab
                key={tab.id}
                label={
                  <Box 
                    sx={{ 
                      display: 'flex', 
                      alignItems: 'center',
                      textTransform: 'none',
                      minWidth: 0,
                      maxWidth: '150px'
                    }}
                  >
                    <Box 
                      component="span" 
                      sx={{ 
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        display: 'block'
                      }}
                    >
                      {tab.title}
                    </Box>
                  </Box>
                }
                sx={{
                  minHeight: '48px',
                  minWidth: '80px',
                  maxWidth: '200px',
                  p: 0,
                  '&.Mui-selected': {
                    bgcolor: 'background.paper',
                    borderRight: '1px solid',
                    borderLeft: '1px solid',
                    borderColor: 'divider',
                    borderTop: '2px solid',
                    borderTopColor: 'primary.main',
                    ml: -0.5,
                    mr: -0.5
                  }
                }}
              />
            ))}
          </Tabs>
        </Box>

        {/* Right arrow */}
        <IconButton 
          size="small" 
          disabled={tabValue >= tabs.length - 1}
          onClick={() => setTabValue(prev => Math.min(tabs.length - 1, prev + 1))}
          sx={{ 
            mr: 0.5,
            my: 'auto',
            visibility: tabs.length > 1 ? 'visible' : 'hidden'
          }}
        >
          <ArrowForwardIosIcon fontSize="small" />
        </IconButton>

        {/* Add tab button */}
        <Tooltip title="New Tab">
          <IconButton 
            size="small" 
            onClick={handleAddTab}
            sx={{ 
              mr: 1,
              my: 'auto'
            }}
          >
            <AddIcon fontSize="small" />
          </IconButton>
        </Tooltip>

        {/* Collapse button - positioned outside on the left */}
        <IconButton 
          className="collapse-button"
          size="small" 
          onClick={onClose}
          sx={{
            position: 'absolute',
            left: '-24px',
            top: '50%',
            transform: 'translateY(-50%)',
            bgcolor: 'background.paper',
            border: '1px solid',
            borderRight: 'none',
            borderColor: 'divider',
            borderTopLeftRadius: '4px',
            borderBottomLeftRadius: '4px',
            height: '60px',
            width: '24px',
            opacity: 0.7,
            transition: 'opacity 0.2s ease',
            '&:hover': {
              bgcolor: 'action.hover',
              opacity: 1
            }
          }}
        >
          <ChevronRightIcon fontSize="small" />
        </IconButton>
      </Box>

      {/* Tab content */}
      <Box sx={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {/* Search bar */}
        <Paper
          component="form"
          sx={{ 
            p: '2px 4px', 
            display: 'flex', 
            alignItems: 'center',
            m: 1,
            borderRadius: 2
          }}
        >
          <IconButton sx={{ p: '10px' }} aria-label="search">
            <SearchIcon />
          </IconButton>
          <InputBase
            sx={{ ml: 1, flex: 1 }}
            placeholder="Search nodes..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          {searchQuery && (
            <IconButton 
              onClick={() => setSearchQuery('')}
              sx={{ p: '10px' }}
            >
              <CloseIcon fontSize="small" />
            </IconButton>
          )}
        </Paper>

        {/* Tab panels */}
        <Box sx={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          {tabs.map((tab, index) => (
            <TabPanel 
              key={tab.id} 
              value={tabValue} 
              index={index}
              sx={{ flex: 1, overflow: 'hidden' }}
            >
              <NodeDetailsTab node={tab.node} />
            </TabPanel>
          ))}
        </Box>
      </Box>
    </Box>
  );
};

export default SidePanel;

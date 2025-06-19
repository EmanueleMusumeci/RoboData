import React, { useState } from 'react';
import { 
  AppBar, 
  Box, 
  Button, 
  CssBaseline, 
  Drawer, 
  IconButton, 
  List, 
  ListItem, 
  ListItemText, 
  TextField, 
  Toolbar, 
  Typography,
  Paper,
  Divider,
  useTheme,
  useMediaQuery
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';  
import GraphVisualization from './components/GraphVisualization';
import SidePanel from './components/SidePanel';
import './App.css';

const drawerWidth = 280;
const sidePanelWidth = 400;

function App() {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [mobileOpen, setMobileOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [selectedNode, setSelectedNode] = useState(null);
  const [isSidePanelOpen, setIsSidePanelOpen] = useState(false);
  
  // Handle node click from GraphVisualization
  const handleNodeClick = (event, node) => {
    event.stopPropagation();
    setSelectedNode(node);
    if (isMobile) {
      setIsSidePanelOpen(true);
    } else if (!isSidePanelOpen) {
      setIsSidePanelOpen(true);
    }
  };

  const handleCloseSidePanel = () => {
    setIsSidePanelOpen(false);
    setSelectedNode(null);
  };

  const handleBackdropClick = () => {
    if (isMobile) {
      setIsSidePanelOpen(false);
    }
  };
  const [graphData] = useState({
    nodes: [
      { 
        id: 'Q1', 
        label: 'Entity 1', 
        description: 'This is a sample entity with some properties',
        color: '#1976d2', 
        borderColor: '#0d47a1',
        properties: [
          { id: 'P31', label: 'instance of', value: 'human' },
          { id: 'P21', label: 'sex or gender', value: 'male' },
        ]
      },
      { 
        id: 'Q2', 
        label: 'Entity 2', 
        description: 'Another sample entity with different properties',
        color: '#d32f2f', 
        borderColor: '#b71c1c',
        properties: [
          { id: 'P31', label: 'instance of', value: 'human' },
          { id: 'P21', label: 'sex or gender', value: 'female' },
        ]
      },
      { 
        id: 'Q3', 
        label: 'Entity 3', 
        description: 'Yet another sample entity',
        color: '#388e3c', 
        borderColor: '#1b5e20',
        properties: [
          { id: 'P31', label: 'instance of', value: 'organization' },
        ]
      },
    ],
    links: [
      { source: 'Q1', target: 'Q2', label: 'knows' },
      { source: 'Q2', target: 'Q3', label: 'works at' },
    ]
  });

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleQuerySubmit = (e) => {
    e.preventDefault();
    // TODO: Implement query submission
    console.log('Query submitted:', query);
  };

  const drawer = (
    <div>
      <Toolbar>
        <Typography variant="h6" noWrap component="div">
          RoboData Controls
        </Typography>
      </Toolbar>
      <Divider />
      <Box sx={{ p: 2 }}>
        <form onSubmit={handleQuerySubmit}>
          <TextField
            fullWidth
            variant="outlined"
            label="Enter your query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            margin="normal"
          />
          <Button 
            type="submit" 
            variant="contained" 
            color="primary" 
            fullWidth
            sx={{ mt: 2 }}
          >
            Search
          </Button>
        </form>
        
        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            Recent Queries
          </Typography>
          <List>
            <ListItem button>
              <ListItemText primary="Show properties of Q1" />
            </ListItem>
            <ListItem button>
              <ListItemText primary="Find path between Q1 and Q3" />
            </ListItem>
          </List>
        </Box>
      </Box>
    </div>
  );

  return (
    <Box sx={{ display: 'flex', position: 'relative' }}>
      <CssBaseline />
      
      {/* Side Panel Overlay for Mobile */}
      {isMobile && isSidePanelOpen && (
        <Box
          sx={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            zIndex: 1200,
          }}
          onClick={handleBackdropClick}
        />
      )}

      <AppBar
        position="fixed"
        sx={{
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          ml: { sm: `${drawerWidth}px` },
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div">
            RoboData Knowledge Graph
          </Typography>
        </Toolbar>
      </AppBar>
      <Box
        component="nav"
        sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
        aria-label="mailbox folders"
      >
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better open performance on mobile.
          }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
        >
          {drawer}
        </Drawer>
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>
      <Box sx={{ display: 'flex', width: '100%' }}>
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            p: 3,
            width: { 
              sm: isSidePanelOpen 
                ? `calc(100% - ${drawerWidth}px - ${sidePanelWidth}px)`
                : `calc(100% - ${drawerWidth}px)`
            },
            transition: theme.transitions.create(['width', 'margin'], {
              easing: theme.transitions.easing.sharp,
              duration: theme.transitions.duration.leavingScreen,
            }),
            ...(isSidePanelOpen && {
              transition: theme.transitions.create(['width', 'margin'], {
                easing: theme.transitions.easing.easeOut,
                duration: theme.transitions.duration.enteringScreen,
              }),
            }),
          }}
        >
          <Toolbar />
          <Paper sx={{ p: 2, height: 'calc(100vh - 100px)' }}>
            <GraphVisualization 
              data={graphData} 
              onNodeClick={handleNodeClick}
              isSidePanelOpen={isSidePanelOpen}
              selectedNodeId={selectedNode?.id}
            />
          </Paper>
        </Box>

        {/* Side Panel */}
        <Box
          sx={{
            width: sidePanelWidth,
            flexShrink: 0,
            height: '100vh',
            position: 'relative',
            display: isSidePanelOpen ? 'block' : 'none',
            [theme.breakpoints.down('sm')]: {
              position: 'fixed',
              right: 0,
              top: 0,
              zIndex: 1300,
              display: isSidePanelOpen ? 'block' : 'none',
            },
          }}
        >
          <SidePanel
            isOpen={isSidePanelOpen}
            onClose={handleCloseSidePanel}
            selectedNode={selectedNode}
            nodes={graphData.nodes}
            isMobile={isMobile}
          />
        </Box>
      </Box>
    </Box>
  );
}

export default App;

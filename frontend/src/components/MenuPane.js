import React, { useState } from 'react';
import { Box, Button, TextField, Typography } from '@mui/material';

const MenuPane = ({ entityId, onEntityChange, onAction }) => {
  const [currentEntity, setCurrentEntity] = useState(entityId);

  const handleEntityChange = (e) => {
    const newEntity = e.target.value;
    setCurrentEntity(newEntity);
    onEntityChange(newEntity);
  };

  const handleAction = (actionType) => {
    onAction({
      type: actionType,
      entity: currentEntity
    });
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Menu
      </Typography>

      <Box sx={{ mb: 3 }}>
        <TextField
          fullWidth
          label="Wikidata Entity ID"
          value={currentEntity}
          onChange={handleEntityChange}
          variant="outlined"
          sx={{ mb: 2 }}
        />
      </Box>

      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <Button
          variant="contained"
          onClick={() => handleAction('inspect')}
        >
          Inspect Properties
        </Button>
        <Button
          variant="contained"
          onClick={() => handleAction('neighbors')}
        >
          List Neighbors
        </Button>
        <Button
          variant="contained"
          onClick={() => handleAction('query')}
        >
          Query
        </Button>
      </Box>
    </Box>
  );
};

export default MenuPane;

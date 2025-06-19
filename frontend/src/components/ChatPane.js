import React, { useState } from 'react';
import { Box, Paper, TextField, IconButton, Typography } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';

const ChatPane = ({ messages, onSend }) => {
  const [message, setMessage] = useState('');

  const handleSend = () => {
    if (message.trim()) {
      onSend(message);
      setMessage('');
    }
  };

  return (
    <Paper sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ flexGrow: 1, overflow: 'auto', mb: 2 }}>
        {messages.map((msg, index) => (
          <Box
            key={index}
            sx={{
              p: 2,
              mb: 2,
              borderRadius: 2,
              bgcolor: msg.isUser ? 'primary.light' : 'grey.100',
              alignSelf: msg.isUser ? 'flex-end' : 'flex-start',
              maxWidth: '80%'
            }}
          >
            <Typography>{msg.content}</Typography>
          </Box>
        ))}
      </Box>

      <Box sx={{ display: 'flex', gap: 1 }}>
        <TextField
          fullWidth
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type your message..."
          variant="outlined"
          sx={{ flexGrow: 1 }}
        />
        <IconButton onClick={handleSend}>
          <SendIcon />
        </IconButton>
      </Box>
    </Paper>
  );
};

export default ChatPane;

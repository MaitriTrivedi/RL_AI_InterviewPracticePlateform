import React from 'react';
import { Box, Container } from '@mui/material';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <Box
      sx={{
        minHeight: '100vh',
        backgroundColor: 'background.default',
        py: 4,
      }}
    >
      <Container 
        maxWidth="lg"
        sx={{
          backgroundColor: 'background.paper',
          borderRadius: 2,
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
          p: 4,
        }}
      >
        {children}
      </Container>
    </Box>
  );
};

export default Layout; 
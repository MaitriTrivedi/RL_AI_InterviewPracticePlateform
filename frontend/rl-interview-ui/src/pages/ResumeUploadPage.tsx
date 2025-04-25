import React, { useState, useRef } from 'react';
import { 
  Box, 
  Button, 
  Typography, 
  Paper, 
  Card, 
  CardContent, 
  TextField, 
  Divider,
  Alert,
  AlertTitle,
  Chip,
  LinearProgress
} from '@mui/material';
import { 
  CloudUpload as CloudUploadIcon,
  PictureAsPdf as PdfIcon, 
  Delete as DeleteIcon,
  CheckCircleOutline as CheckIcon
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useAppContext } from '../context/AppContext';
import { resumeApi } from '../services/api';
import LoadingSpinner from '../components/common/LoadingSpinner';
import Layout from '../components/layout/Layout';

const ResumeUploadPage: React.FC = () => {
  const navigate = useNavigate();
  const { state, setLoading, setError, setResumeId, setCurrentResume } = useAppContext();
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  
  // Handle drag events
  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(true);
  };
  
  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
  };
  
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type === 'application/pdf') {
        setFile(droppedFile);
        setError(null);
      } else {
        setError('Please upload a PDF file');
      }
    }
  };
  
  // Handle file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      if (selectedFile.type === 'application/pdf') {
        setFile(selectedFile);
        setError(null);
      } else {
        setError('Please upload a PDF file');
        setFile(null);
      }
    }
  };
  
  // Handle file upload button click
  const handleSelectClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };
  
  // Handle removing the selected file
  const handleRemoveFile = () => {
    setFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    setUploadSuccess(false);
  };
  
  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!file) {
      setError('Please select a resume file to upload');
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      
      // Simulate progress for better UX
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return prev;
          }
          return prev + 10;
        });
      }, 500);
      
      // Upload the file
      const response = await resumeApi.uploadResume(file);
      
      clearInterval(progressInterval);
      setUploadProgress(100);
      
      // Check response
      if (response.data && response.data.resumeId) {
        setResumeId(response.data.resumeId);
        setCurrentResume(response.data.resumeData);
        setUploadSuccess(true);
        
        // Small delay to show 100% progress
        setTimeout(() => {
          setUploadProgress(0);
        }, 500);
      } else {
        throw new Error('Invalid response from server');
      }
    } catch (err: any) {
      setError(err.message || 'Failed to upload resume');
      setUploadProgress(0);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <Layout>
      <Box>
        <Typography variant="h4" component="h1" gutterBottom>
          Resume Upload
        </Typography>
        
        <Typography variant="body1" paragraph>
          Upload your resume in PDF format. We'll analyze it to generate relevant interview questions 
          tailored to your skills and experience.
        </Typography>
        
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <form onSubmit={handleSubmit}>
              {/* Upload Area */}
              <Box
                sx={{
                  border: '2px dashed',
                  borderColor: dragging ? 'primary.main' : 'grey.400',
                  borderRadius: 2,
                  p: 3,
                  textAlign: 'center',
                  bgcolor: dragging ? 'primary.light' : 'background.paper',
                  transition: 'all 0.3s',
                  mb: 3,
                  cursor: 'pointer',
                }}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={handleSelectClick}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  accept=".pdf"
                  style={{ display: 'none' }}
                />
                
                <CloudUploadIcon 
                  sx={{ 
                    fontSize: 60, 
                    color: dragging ? 'primary.dark' : 'text.secondary',
                    mb: 2 
                  }} 
                />
                
                <Typography variant="h6" gutterBottom>
                  {dragging 
                    ? 'Drop your resume here' 
                    : 'Drag & drop your resume or click to browse'}
                </Typography>
                
                <Typography variant="body2" color="text.secondary">
                  Supports PDF format only
                </Typography>
              </Box>
              
              {/* Selected File */}
              {file && (
                <Box>
                  <Paper 
                    variant="outlined" 
                    sx={{ 
                      p: 2, 
                      display: 'flex', 
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      mb: 3
                    }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <PdfIcon color="error" sx={{ mr: 1 }} />
                      <Box>
                        <Typography variant="body1">{file.name}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          {(file.size / 1024).toFixed(1)} KB
                        </Typography>
                      </Box>
                    </Box>
                    <Button 
                      startIcon={<DeleteIcon />}
                      color="error"
                      onClick={handleRemoveFile}
                      variant="outlined"
                    >
                      Remove
                    </Button>
                  </Paper>
                  
                  {/* Upload Progress */}
                  {uploadProgress > 0 && (
                    <Box sx={{ mb: 3 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <Typography variant="body2" sx={{ mr: 1 }}>
                          Uploading...
                        </Typography>
                        <Typography variant="body2" color="primary" fontWeight="bold">
                          {uploadProgress}%
                        </Typography>
                      </Box>
                      <LinearProgress 
                        variant="determinate" 
                        value={uploadProgress} 
                        sx={{ height: 8, borderRadius: 2 }}
                      />
                    </Box>
                  )}
                </Box>
              )}
              
              {/* Success Message */}
              {uploadSuccess && (
                <Alert 
                  severity="success" 
                  sx={{ mb: 3 }}
                  icon={<CheckIcon fontSize="inherit" />}
                >
                  <AlertTitle>Resume Uploaded Successfully</AlertTitle>
                  Your resume has been uploaded and processed. You can now proceed to start the interview.
                </Alert>
              )}
              
              {/* Submit Button */}
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
                <Button 
                  type="submit" 
                  variant="contained" 
                  color="primary"
                  size="large"
                  disabled={!file || state.loading || uploadSuccess}
                  startIcon={<CloudUploadIcon />}
                >
                  {state.loading ? 'Uploading...' : 'Upload Resume'}
                </Button>
                
                {uploadSuccess && (
                  <Button 
                    variant="contained" 
                    color="secondary"
                    size="large"
                    onClick={() => navigate('/interview')}
                  >
                    Continue to Interview
                  </Button>
                )}
              </Box>
            </form>
          </CardContent>
        </Card>
        
        {/* Resume Tips */}
        <Typography variant="h5" component="h2" gutterBottom sx={{ mt: 4 }}>
          Resume Tips for Better Results
        </Typography>
        
        <Paper sx={{ p: 3 }}>
          <Typography variant="body1" paragraph>
            To get the most relevant interview questions, make sure your resume:
          </Typography>
          
          <Box sx={{ mb: 2 }}>
            <Chip label="Clear Format" color="primary" size="small" sx={{ mr: 1, mb: 1 }} />
            <Chip label="PDF Format" color="primary" size="small" sx={{ mr: 1, mb: 1 }} />
            <Chip label="Skills Section" color="primary" size="small" sx={{ mr: 1, mb: 1 }} />
            <Chip label="Work History" color="primary" size="small" sx={{ mr: 1, mb: 1 }} />
            <Chip label="Education" color="primary" size="small" sx={{ mr: 1, mb: 1 }} />
            <Chip label="Projects" color="primary" size="small" sx={{ mr: 1, mb: 1 }} />
          </Box>
          
          <Divider sx={{ my: 2 }} />
          
          <Typography variant="body2" color="text.secondary" paragraph>
            Our system uses advanced natural language processing to extract information from your resume.
            The more detailed and structured your resume is, the better our AI can generate relevant questions.
          </Typography>
        </Paper>
        
        {/* Loading Overlay */}
        {state.loading && (
          <Box
            sx={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              bgcolor: 'rgba(255, 255, 255, 0.7)',
              zIndex: 1300,
            }}
          >
            <LoadingSpinner message="Processing your resume..." />
          </Box>
        )}
      </Box>
    </Layout>
  );
};

export default ResumeUploadPage; 
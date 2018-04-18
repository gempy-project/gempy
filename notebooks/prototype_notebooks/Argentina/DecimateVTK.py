#!python

import vtk

fin='Crust.xyz'
fran = vtk.vtkPolyDataReader()
fran.SetFileName(fin)

deci = vtk.vtkDecimatePro()
deci.SetInputConnection(fran.GetOutputPort())

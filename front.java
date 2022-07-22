package abhinav;
import java.sql.*;
import java.util.*;
import java.awt.*;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
public class front extends JFrame implements ActionListener
{
	 
	
	Connection con=null;
	PreparedStatement ps=null;
	front() throws SQLException
	{
		
		
		JFrame frame=new JFrame("Task Management System");

		frame.setVisible(true);
		frame.getContentPane().setBackground(new Color(224, 255, 255));
	
		 frame.setBounds(10,10,1000,600);
		 frame.setLayout(null);
		 
		JLabel l1=new JLabel("TASK MANAGEMENT SYSTEM");
		l1.setFont(new Font("Arial Black", Font.BOLD, 25)); 
		
		l1.setForeground(Color.BLUE);
       
        JLabel l2=new JLabel("Enter Task :");
        JLabel l3=new JLabel("Record to Delete :");
        l2.setFont(new Font("Arial Black", Font.BOLD, 15));
        l3.setFont(new Font("Arial Black", Font.BOLD, 15));
        l1.setBounds(450,0,1000,70);
		l2.setBounds(100,150,200,100);
		l3.setBounds(100,200,300,100);
		
		
		 
		JTextField textfield1=new JTextField(20);
		
		textfield1.setBounds(300,190,200,20);

		
		JTextField textfield2=new JTextField(20);
		
		textfield2.setBounds(300,240,200,20);
		
		
		JButton button1=new JButton("ADD>>");
		button1.setBounds(135,380,90,30);
	
		JButton button2=new JButton("Show");
		button2.setBounds(235,380,90,30);
		
		JButton button3=new JButton("Delete");
		button3.setBounds(335,380,90,30);
		
		JButton button4=new JButton("Done");
		button4.setBounds(435,380,90,30);
		
		JTable table = new JTable();
		table.setGridColor(Color.ORANGE);
		table.setCellSelectionEnabled(false);
		JScrollPane jp=new JScrollPane(table);
		jp.setBounds(700,120,300,300);

		frame.setVisible(true);
		
		frame.add(l1);
		frame.add(l2);
		frame.add(l3);
		frame.add(textfield1);
		frame.add(textfield2);
		frame.add(button1);
		frame.add(button2);
		frame.add(button3);
		frame.add(jp);
		frame.add(button4);
		con=DriverManager.getConnection("jdbc:mysql://localhost:3306/Task","root","root");
		
		button1.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e)
			{
				
				String query=e.getActionCommand();
				if(query.equals("ADD>>"))
				{
					try
					{
						if(textfield1.getText().equals(""))
						{
							JOptionPane.showMessageDialog(frame,"Fields Cannot Be Empty!!");
						}
						else
						{
							String s1=textfield1.getText();
							
							
							ps=(PreparedStatement)con.prepareStatement("insert into task(Task) values(?)");
							ps.setString(1,s1);
							
							ps.executeUpdate();
					
							textfield1.setText("");
							JOptionPane.showMessageDialog(frame,"Task Added Successfully");
						}
					
					}
					catch(Exception f)
					{
		
					}
				}
			}
		});
		
		button2.addActionListener(new ActionListener()
		{
			public void actionPerformed(ActionEvent e4) 
			{
				String query=e4.getActionCommand();
				if(query.equals("Show"))
				{
					
					try
					{
						 PreparedStatement pst=con.prepareStatement("select * from task");
						ResultSet rs=pst.executeQuery();
						ResultSetMetaData rd=rs.getMetaData();
						DefaultTableModel model=(DefaultTableModel)table.getModel();
						int a=rd.getColumnCount();
						model.setRowCount(0);
						String[] col=new String[a];
				
				
						for(int i=0;i<a;i++)
						{
							col[i]=rd.getColumnName(i+1);
						}
				
						model.setColumnIdentifiers(col);
						
						String imp;
				
						while(rs.next())
						{
					
							imp=rs.getString(1);
							
							String[] row= {imp};
							model.addRow(row);
						}
				
					}
					catch(Exception f)
					{
						System.out.println(f);
					}
				}
			}	
		});
		button3.addActionListener(new ActionListener()
		{
			public void actionPerformed(ActionEvent e3)
			{
				String query=e3.getActionCommand();
				if(query.equals("Delete")) 
				{
					String val=textfield2.getText();
					
					try
					{
						 PreparedStatement ps1=con.prepareStatement("select * from task");
						ResultSet result=ps1.executeQuery();
						int a=0;
						while(result.next())
						{
							String val1=result.getString(1);
							
							if(val1.equals(val))
							{
								a++;
								PreparedStatement pst=con.prepareStatement("delete from task where Task=?");
								pst.setString(1,val);
								pst.executeUpdate();
							}	
						}
					    if(a==0)
						{
							JOptionPane.showMessageDialog(frame,"No Such Records Found !!");
						}
						else
						{
							JOptionPane.showMessageDialog(frame,"Deleted Successfully !!");
						}
						
						textfield2.setText("");
						
					}
					catch(Exception e)
					{
						System.out.println(e);
					}
					
				}
			}
				
		});
		
		
		button4.addActionListener(new ActionListener()
		{
			public void actionPerformed(ActionEvent e3) {
				String query=e3.getActionCommand();
				if(query.equals("Done"))
				{
					
					int result=JOptionPane.showConfirmDialog(frame,"Do You Want to Exit !!","Confirm",JOptionPane.YES_NO_OPTION,JOptionPane.QUESTION_MESSAGE);
					if(result==JOptionPane.YES_OPTION)
					{
						try {
							con.close();
						}
						catch(Exception a)
						{
							
						}
						System.exit(0);
					}
				}
				}
				
		});
		
		
		
	}
	public static void main(String args[])throws SQLException
	{	
		new front();
		try
		{
			Class.forName("com.mysql.cj.jdbc.Driver");
		}
		catch(Exception g)
		{
			System.out.println(g);
		}
		
	}
	
}
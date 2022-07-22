package abhinav;
import java.awt.*;
import java.sql.*;
import java.awt.event.*;
import javax.swing.*;
public class login {
	login() 
	{
		JFrame f1=new JFrame("Login Page");
		f1.setSize(500,600);
		f1.setVisible(true);
		f1.setLayout(null);
		
		JLabel l1=new JLabel("Login");
		l1.setFont(new Font("Arial Black", Font.BOLD, 25)); 
		l1.setBounds(200,50,200,30);
		
		JLabel l2=new JLabel("UserName");
		l2.setFont(new Font("Arial Black", Font.BOLD, 15)); 
		l2.setBounds(60,150,200,30);
		
		JLabel l3=new JLabel("Password");
		l3.setFont(new Font("Arial Black", Font.BOLD, 15)); 
		l3.setBounds(60,200,200,30);
		
		JTextField tf1=new JTextField(20);
		tf1.setBounds(165,155,200,20);
		
		JPasswordField value=new JPasswordField();
		value.setBounds(165,205,200,20);
		
		JButton jb=new JButton("Login");
		
		jb.setBounds(200,280,90,20);
		f1.add(jb);
		f1.add(l1);
		f1.add(l2);
		f1.add(l3);
		f1.add(tf1);
		f1.add(value);
	
		jb.addActionListener(new ActionListener()
		{
			public void actionPerformed(ActionEvent e4) {
				String s4=e4.getActionCommand();
				if(s4.equals("Login"))
				{
					String s1=new String(value.getPassword());
					
					if(tf1.getText().equals("abhinav") && s1.equals("12345"))
					{
						try {
						new front();
						}
						catch(Exception e)
						{
							System.out.println(e);
						}
					}
					else
					{
						JOptionPane.showMessageDialog(f1,"Invalid UserName or Password");
					}
				}
				}
				
		});
		
	}
	public static void main(String args[]) throws SQLException
	{
		new login();
	}
}